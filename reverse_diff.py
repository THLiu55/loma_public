import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff
import string
import random

def Log(title, msg):
    print("******************")
    print(f"*** {title}")
    print(msg)
    print("******************")

# From https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
def random_id_generator(size=0, chars=string.ascii_lowercase + string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def reverse_diff(diff_func_id : str,
                 structs : dict[str, loma_ir.Struct],
                 funcs : dict[str, loma_ir.func],
                 diff_structs : dict[str, loma_ir.Struct],
                 func : loma_ir.FunctionDef,
                 func_to_rev : dict[str, str]) -> loma_ir.FunctionDef:
    """ Given a primal loma function func, apply reverse differentiation
        and return a function that computes the total derivative of func.

        For example, given the following function:
        def square(x : In[float]) -> float:
            return x * x
        and let diff_func_id = 'd_square', reverse_diff() should return
        def d_square(x : In[float], _dx : Out[float], _dreturn : float):
            _dx = _dx + _dreturn * x + _dreturn * x

        Parameters:
        diff_func_id - the ID of the returned function
        structs - a dictionary that maps the ID of a Struct to 
                the corresponding Struct
        funcs - a dictionary that maps the ID of a function to 
                the corresponding func
        diff_structs - a dictionary that maps the ID of the primal
                Struct to the corresponding differential Struct
                e.g., diff_structs['float'] returns _dfloat
        func - the function to be differentiated
        func_to_rev - mapping from primal function ID to its reverse differentiation
    """

    # Some utility functions you can use for your homework.
    def type_to_string(t):
        match t:
            case loma_ir.Int():
                return 'int'
            case loma_ir.Float():
                return 'float'
            case loma_ir.Array():
                return 'array_' + type_to_string(t.t)
            case loma_ir.Struct():
                return t.id
            case _:
                assert False

    def var_to_differential(expr, var_to_dvar):
        match expr:
            case loma_ir.Var():
                return loma_ir.Var(var_to_dvar[expr.id], t = expr.t)
            case loma_ir.ArrayAccess():
                return loma_ir.ArrayAccess(\
                    var_to_differential(expr.array, var_to_dvar),
                    expr.index,
                    t = expr.t)
            case loma_ir.StructAccess():
                return loma_ir.StructAccess(\
                    var_to_differential(expr.struct, var_to_dvar),
                    expr.member_id,
                    t = expr.t)
            case _:
                assert False

    def assign_zero(target):
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                return [loma_ir.Assign(target, loma_ir.ConstFloat(0.0))]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(
                        target, m.id, t = m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += assign_zero(target_m)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += assign_zero(target_m)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t = m.t.t)
                            stmts += assign_zero(target_m)
                return stmts
            case loma_ir.Array():
                assert target.static_size is not None
                stmts = []

            case _:
                assert False

    def accum_deriv(target, deriv, overwrite, simd=False):
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                if overwrite:
                    return [loma_ir.Assign(target, deriv)]
                else:
                    if simd:
                        return [loma_ir.CallStmt(loma_ir.Call('atomic_add', [target, deriv]))]
                    return [loma_ir.Assign(target,
                        loma_ir.BinaryOp(loma_ir.Add(), target, deriv))]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(
                        target, m.id, t = m.t)
                    deriv_m = loma_ir.StructAccess(
                        deriv, m.id, t = m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += accum_deriv(target_m, deriv_m, overwrite)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += accum_deriv(target_m, deriv_m, overwrite)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t = m.t.t)
                            deriv_m = loma_ir.ArrayAccess(
                                deriv_m, loma_ir.ConstInt(i), t = m.t.t)
                            stmts += accum_deriv(target_m, deriv_m, overwrite)
                return stmts
            case _:
                assert False

    def check_lhs_is_output_arg(lhs, output_args):
        match lhs:
            case loma_ir.Var():
                return lhs.id in output_args
            case loma_ir.StructAccess():
                return check_lhs_is_output_arg(lhs.struct, output_args)
            case loma_ir.ArrayAccess():
                return check_lhs_is_output_arg(lhs.array, output_args)
            case _:
                assert False

    # A utility class that you can use for HW3.
    # This mutator normalizes each call expression into
    # f(x0, x1, ...)
    # where x0, x1, ... are all loma_ir.Var or 
    # loma_ir.ArrayAccess or loma_ir.StructAccess
    # Furthermore, it normalizes all Assign statements
    # with a function call
    # z = f(...)
    # into a declaration followed by an assignment
    # _tmp : [z's type]
    # _tmp = f(...)
    # z = _tmp
    class CallNormalizeMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            self.tmp_count = 0
            self.tmp_declare_stmts = []
            new_body = [self.mutate_stmt(stmt) for stmt in node.body]
            new_body = irmutator.flatten(new_body)

            new_body = self.tmp_declare_stmts + new_body

            return loma_ir.FunctionDef(\
                node.id, node.args, new_body, node.is_simd, node.ret_type, lineno = node.lineno)

        def mutate_return(self, node):
            self.tmp_assign_stmts = []
            val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Return(\
                val,
                lineno = node.lineno)]

        def mutate_declare(self, node):
            self.tmp_assign_stmts = []
            val = None
            if node.val is not None:
                val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Declare(\
                node.target,
                node.t,
                val,
                lineno = node.lineno)]

        def mutate_assign(self, node):
            self.tmp_assign_stmts = []
            target = self.mutate_expr(node.target)
            self.has_call_expr = False
            val = self.mutate_expr(node.val)
            if self.has_call_expr:
                # turn the assignment into a declaration plus
                # an assignment
                self.tmp_count += 1
                tmp_name = f'_call_t_{self.tmp_count}_{random_id_generator()}'
                self.tmp_count += 1
                self.tmp_declare_stmts.append(loma_ir.Declare(\
                    tmp_name,
                    target.t,
                    lineno = node.lineno))
                tmp_var = loma_ir.Var(tmp_name, t = target.t)
                assign_tmp = loma_ir.Assign(\
                    tmp_var,
                    val,
                    lineno = node.lineno)
                assign_target = loma_ir.Assign(\
                    target,
                    tmp_var,
                    lineno = node.lineno)
                return self.tmp_assign_stmts + [assign_tmp, assign_target]
            else:
                return self.tmp_assign_stmts + [loma_ir.Assign(\
                    target,
                    val,
                    lineno = node.lineno)]

        def mutate_call_stmt(self, node):
            self.tmp_assign_stmts = []
            call = self.mutate_expr(node.call)
            return self.tmp_assign_stmts + [loma_ir.CallStmt(\
                call,
                lineno = node.lineno)]

        def mutate_call(self, node):
            self.has_call_expr = True
            new_args = []
            for arg in node.args:
                if not isinstance(arg, loma_ir.Var) and \
                        not isinstance(arg, loma_ir.ArrayAccess) and \
                        not isinstance(arg, loma_ir.StructAccess):
                    arg = self.mutate_expr(arg)
                    tmp_name = f'_call_t_{self.tmp_count}_{random_id_generator()}'
                    self.tmp_count += 1
                    tmp_var = loma_ir.Var(tmp_name, t = arg.t)
                    self.tmp_declare_stmts.append(loma_ir.Declare(\
                        tmp_name, arg.t))
                    self.tmp_assign_stmts.append(loma_ir.Assign(\
                        tmp_var, arg))
                    new_args.append(tmp_var)
                else:
                    new_args.append(arg)
            return loma_ir.Call(node.id, new_args, t = node.t)

        def mutate_while(self, node):
            new_body = []
            for stmt in node.body:
                if not isinstance(stmt, loma_ir.Assign):
                    new_body.append(self.mutate_stmt(stmt))
                else:
                    new_body.append(stmt)
            return loma_ir.While(node.cond, node.max_iter,
                new_body, lineno = node.lineno)


    class ForwardPassMutator(irmutator.IRMutator):
        def __init__(self, output_args):
            self.output_args = output_args
            self.cache_vars_list = {}
            self.var_to_dvar = {}
            self.type_cache_size = {}
            self.type_to_stack_and_ptr_names = {}

        def mutate_return(self, node):
            return []

        def mutate_declare(self, node):
            # For each declaration, add another declaration for the derivatives
            # except when it's an integer
            if node.t != loma_ir.Int():
                dvar = '_d' + node.target + '_' + random_id_generator()
                self.var_to_dvar[node.target] = dvar
                return [node, loma_ir.Declare(\
                    dvar,
                    node.t,
                    lineno = node.lineno)]
            else:
                return node

        def mutate_assign(self, node, stack_pos=1): 
            if not hasattr(self, 'stack_size_map'):
                print("create a new map")
                self.stack_size_map = {}
            if check_lhs_is_output_arg(node.target, self.output_args):
                return []

            # y = f(x0, x1, ..., y)
            # we will use a temporary array _t to hold variable y for later use:
            # _t[stack_pos++] = y
            # y = f(x0, x1, ..., y)
            assign_primal = loma_ir.Assign(\
                node.target,
                self.mutate_expr(node.val),
                lineno = node.lineno)
            # backup
            t_str = type_to_string(node.val.t)
            if t_str in self.type_to_stack_and_ptr_names:
                stack_name, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
            else:
                random_id = random_id_generator()
                stack_name = f'_t_{t_str}_{random_id}'
                stack_ptr_name = f'_stack_ptr_{t_str}_{random_id}'
                self.type_to_stack_and_ptr_names[t_str] = (stack_name, stack_ptr_name)
            
            
            if stack_name in self.stack_size_map:
                self.stack_size_map[stack_name] += stack_pos
            else:
                self.stack_size_map[stack_name] = stack_pos
            
            stack_ptr_var = loma_ir.Var(stack_ptr_name, t=loma_ir.Int())
            cache_var_expr = loma_ir.ArrayAccess(
                loma_ir.Var(stack_name),
                stack_ptr_var,
                t = node.val.t)
            cache_primal = loma_ir.Assign(cache_var_expr, node.target)
            stack_advance = loma_ir.Assign(stack_ptr_var,
                loma_ir.BinaryOp(loma_ir.Add(), stack_ptr_var, loma_ir.ConstInt(1)))

            if node.val.t in self.cache_vars_list:
                self.cache_vars_list[node.val.t].append((cache_var_expr, node.target))
            else:
                self.cache_vars_list[node.val.t] = [(cache_var_expr, node.target)]
            if node.val.t in self.type_cache_size:
                self.type_cache_size[node.val.t] += 1
            else:
                self.type_cache_size[node.val.t] = 1
            return [cache_primal, stack_advance, assign_primal]
        

        def mutate_call_stmt(self, node):
            if node.call.id == 'atomic_add':
                return []
            original_func = funcs[node.call.id]
            cache_primal_stmts = []
            stack_advance_stmts = []
            for arg, para in zip(original_func.args, node.call.args):
                print("arg", arg)
                print("para", para)
                if isinstance(arg.i, loma_ir.Out):
                    if isinstance(arg.t, loma_ir.Array):
                        continue
                    t_str = type_to_string(arg.t)
                    if t_str in self.type_to_stack_and_ptr_names:
                        stack_name, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
                    else:
                        random_id = random_id_generator()
                        stack_name = f'_t_{t_str}_{random_id}'
                        stack_ptr_name = f'_stack_ptr_{t_str}_{random_id}'
                        self.type_to_stack_and_ptr_names[t_str] = (stack_name, stack_ptr_name)
                    stack_ptr_var = loma_ir.Var(stack_ptr_name, t=loma_ir.Int())
                    cache_var_expr = loma_ir.ArrayAccess(
                        loma_ir.Var(stack_name),
                        stack_ptr_var)
                    cache_primal = loma_ir.Assign(cache_var_expr, loma_ir.Var(para.id))
                    stack_advance = loma_ir.Assign(stack_ptr_var,
                        loma_ir.BinaryOp(loma_ir.Add(), stack_ptr_var, loma_ir.ConstInt(1)))
                    cache_primal_stmts.append(cache_primal)
                    stack_advance_stmts.append(stack_advance)
                    if arg.t in self.cache_vars_list:
                        self.cache_vars_list[arg.t].append((cache_var_expr, loma_ir.Var(para.id)))
                    else:
                        self.cache_vars_list[arg.t] = [(cache_var_expr, loma_ir.Var(para.id))]
                    if arg.t in self.type_cache_size:
                        self.type_cache_size[arg.t] += 1
                    else:
                        self.type_cache_size[arg.t] = 1
            if len(cache_primal_stmts) > 0:
                ret = cache_primal_stmts + stack_advance_stmts + [node]
                return ret
            else:
                return node
            
        
        def mutate_while(self, node):
            if not hasattr(self, 'ctr_map'):
                self.ctr_map = {}
                self.ctr_idx = -1
                self.pre_ctr_n = 0
                self.stack_map = {}
            self.ctr_idx += 1            
            self.ctr_name = f'ctr{self.ctr_idx}'                      
            cur_ctr_name = self.ctr_name
            cur_ctr_id = self.ctr_idx
            old_pre_ctr_n = self.pre_ctr_n
            self.ctr_map[self.ctr_name] = old_pre_ctr_n
            self.pre_ctr_n = self.ctr_map[self.ctr_name] * old_pre_ctr_n if self.ctr_map[self.ctr_name] != 0 else node.max_iter
            
            body_stmts = []
            for stmt in node.body:
                if isinstance(stmt, loma_ir.Assign):
                    body_stmts.append(self.mutate_assign(stmt, self.pre_ctr_n))
                else:    
                    body_stmts.append(self.mutate_stmt(stmt))

            if self.ctr_idx != cur_ctr_id:
                body_stmts += [loma_ir.Assign(
                    loma_ir.Var(self.ctr_name + '_ptr', t = loma_ir.Int()),
                    loma_ir.BinaryOp(loma_ir.Add(), loma_ir.Var(self.ctr_name + '_ptr', t = loma_ir.Int()), loma_ir.ConstInt(1))
                )]
                self.pre_ctr_n = old_pre_ctr_n
                self.ctr_idx = cur_ctr_id
                self.ctr_name = cur_ctr_name

            if self.ctr_map[self.ctr_name] == 0:
                iterVar = loma_ir.Var(cur_ctr_name, t = loma_ir.Int())
            else:
                iterVar = loma_ir.ArrayAccess(loma_ir.Var(self.ctr_name), loma_ir.Var(self.ctr_name + '_ptr'))
            body_stmts += [loma_ir.Assign(iterVar, loma_ir.BinaryOp(loma_ir.Add(), iterVar, loma_ir.ConstInt(1)))]
            body_stmts = irmutator.flatten(body_stmts)
            modified_loop = loma_ir.While(node.cond, node.max_iter, body_stmts, lineno = node.lineno)
            return modified_loop


    # HW2 happens here. Modify the following IR mutators to perform
    # reverse differentiation.
    class RevDiffMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            self.simd = node.is_simd
            node = CallNormalizeMutator().mutate_function_def(node)
            random.seed(hash(node.id))
            # Each input argument is followed by an output (the adjoint)
            # Each output is turned into an input
            # The return value turn into an input
            self.var_to_dvar = {}
            new_args = []
            self.output_args = set()
            for arg in node.args:
                if arg.i == loma_ir.In():
                    new_args.append(arg)
                    dvar_id = '_d' + arg.id + '_' + random_id_generator()
                    new_args.append(loma_ir.Arg(dvar_id, arg.t, i = loma_ir.Out()))
                    self.var_to_dvar[arg.id] = dvar_id
                else:
                    assert arg.i == loma_ir.Out()
                    self.output_args.add(arg.id)
                    new_args.append(loma_ir.Arg(arg.id, arg.t, i = loma_ir.In()))
                    self.var_to_dvar[arg.id] = arg.id
            if node.ret_type is not None:
                self.return_var_id = '_dreturn_' + random_id_generator()
                new_args.append(loma_ir.Arg(self.return_var_id, node.ret_type, i = loma_ir.In()))

            # Forward pass
            fm = ForwardPassMutator(self.output_args)
            forward_body = node.body
            mutated_forward = [fm.mutate_stmt(fwd_stmt) for fwd_stmt in forward_body]
            mutated_forward = irmutator.flatten(mutated_forward)
        
            self.var_to_dvar = self.var_to_dvar | fm.var_to_dvar

            self.cache_vars_list = fm.cache_vars_list
            self.type_cache_size = fm.type_cache_size
            self.type_to_stack_and_ptr_names = fm.type_to_stack_and_ptr_names

            tmp_declares = []
            for t, exprs in fm.cache_vars_list.items():
                t_str = type_to_string(t)
                stack_name, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
                if not hasattr(fm, 'stack_size_map') or stack_name not in fm.stack_size_map:
                    size = len(exprs)
                else:
                    size = fm.stack_size_map[stack_name]
                tmp_declares.append(loma_ir.Declare(stack_name,
                    loma_ir.Array(t, size)))
                tmp_declares.append(loma_ir.Declare(stack_ptr_name,
                    loma_ir.Int(), loma_ir.ConstInt(0)))
            
            if hasattr(fm, 'ctr_map'):
                for ctr_name, size in fm.ctr_map.items():
                    if size == 0:    
                        tmp_declares.append(loma_ir.Declare(\
                            ctr_name, loma_ir.Int(), loma_ir.ConstInt(0)))
                    else:
                        tmp_declares.append(loma_ir.Declare(\
                            ctr_name, loma_ir.Array(loma_ir.Int(), size)))
                        tmp_declares.append(loma_ir.Declare(\
                            ctr_name + '_ptr', loma_ir.Int(), loma_ir.ConstInt(0)))
            
            mutated_forward = tmp_declares + mutated_forward


            # Reverse pass
            self.adj_count = 0
            self.in_assign = False
            self.adj_declaration = []
            reversed_body = [self.mutate_stmt(stmt) for stmt in reversed(node.body)]
            reversed_body = irmutator.flatten(reversed_body)

            return loma_ir.FunctionDef(\
                diff_func_id,
                new_args,
                mutated_forward + self.adj_declaration + reversed_body,
                node.is_simd,
                ret_type = None,
                lineno = node.lineno)

        def mutate_return(self, node):
            # Propagate to each variable used in node.val
            self.adj = loma_ir.Var(self.return_var_id, t = node.val.t)
            return self.mutate_expr(node.val)

        def mutate_declare(self, node):
            if node.val is not None:
                if node.t == loma_ir.Int():
                    return []

                self.adj = loma_ir.Var(self.var_to_dvar[node.target])
                return self.mutate_expr(node.val)
            else:
                return []

        def mutate_assign(self, node): 
            if node.val.t == loma_ir.Int():
                stmts = []
                # restore the previous value of this assignment
                t_str = type_to_string(node.val.t)
                _, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
                stack_ptr_var = loma_ir.Var(stack_ptr_name, t=loma_ir.Int())
                stmts.append(loma_ir.Assign(stack_ptr_var,
                    loma_ir.BinaryOp(loma_ir.Sub(), stack_ptr_var, loma_ir.ConstInt(1))))
                cache_var_expr, cache_target = self.cache_vars_list[node.val.t].pop()
                stmts.append(loma_ir.Assign(cache_target, cache_var_expr))
                return stmts

            self.adj = var_to_differential(node.target, self.var_to_dvar)
            if check_lhs_is_output_arg(node.target, self.output_args):
                # if the lhs is an output argument, then we can safely
                # treat this statement the same as "declare"
                return self.mutate_expr(node.val)
            else:
                stmts = []
                # restore the previous value of this assignment
                t_str = type_to_string(node.val.t)
                _, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
                stack_ptr_var = loma_ir.Var(stack_ptr_name, t=loma_ir.Int())
                stmts.append(loma_ir.Assign(stack_ptr_var,
                    loma_ir.BinaryOp(loma_ir.Sub(), stack_ptr_var, loma_ir.ConstInt(1))))
                cache_var_expr, cache_target = self.cache_vars_list[node.val.t].pop()
                stmts.append(loma_ir.Assign(cache_target, cache_var_expr))
                
                # First pass: accumulate
                self.in_assign = True
                self.adj_accum_stmts = []
                stmts += self.mutate_expr(node.val)
                self.in_assign = False

                # zero the target differential
                stmts += assign_zero(var_to_differential(node.target, self.var_to_dvar))

                # Accumulate the adjoints back to the target locations
                stmts += self.adj_accum_stmts
                return stmts

        def mutate_ifelse(self, node):
            else_stmts = [self.mutate_stmt(stmt) for stmt in reversed(node.else_stmts)]
            else_stmts = irmutator.flatten(else_stmts)
            then_stmts = [self.mutate_stmt(stmt) for stmt in reversed(node.then_stmts)]
            then_stmts = irmutator.flatten(then_stmts)
            return loma_ir.IfElse(node.cond, then_stmts, else_stmts, lineno = node.lineno)

        def mutate_call_stmt(self, node):
            if node.call.id == 'atomic_add':
                target = var_to_differential(node.call.args[1], self.var_to_dvar)
                val = var_to_differential(node.call.args[0], self.var_to_dvar)
                ret = [loma_ir.Assign(target,
                                      loma_ir.BinaryOp(loma_ir.Add(), target, val))]
                
                return ret
            original_func = funcs[node.call.id]
            stmt = []
            for arg in original_func.args:
                if isinstance(arg.i, loma_ir.Out):
                    if isinstance(arg.t, loma_ir.Array):
                        continue
                    t_str = type_to_string(arg.t)
                    _, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
                    stack_ptr_var = loma_ir.Var(stack_ptr_name, t=loma_ir.Int())
                    stmt.append(loma_ir.Assign(stack_ptr_var, 
                        loma_ir.BinaryOp(loma_ir.Sub(), stack_ptr_var, loma_ir.ConstInt(1))))
                    cache_var_expr, cache_target = self.cache_vars_list[arg.t].pop()
                    stmt.append(loma_ir.Assign(cache_target, cache_var_expr))
            
            self.adj_accum_stmts = []
            if len(stmt) > 0:
                self.in_assign = True
                stmt += self.mutate_expr(node.call)
                self.in_assign = False
            else:
                stmt += self.mutate_expr(node.call)


            for arg, para in zip(original_func.args, node.call.args):
                if isinstance(arg.i, loma_ir.Out):
                    if isinstance(arg.t, loma_ir.Array):
                        continue    
                    stmt += assign_zero(var_to_differential(para, self.var_to_dvar))
            
            stmt += self.adj_accum_stmts
            return stmt
        

        def mutate_while(self, node):
            if not hasattr(self, 'ctr_idx'):
                self.ctr_idx = -1
            self.ctr_idx += 1
            cur_ctr_idx = self.ctr_idx
            cur_ctr_name = f'ctr{self.ctr_idx}'

            if cur_ctr_idx == 0:
                cmp_left = loma_ir.Var(cur_ctr_name, t = loma_ir.Int())
            else:
                cmp_left = loma_ir.ArrayAccess(loma_ir.Var(cur_ctr_name), loma_ir.Var(cur_ctr_name + '_ptr'))
            new_cond = loma_ir.BinaryOp(loma_ir.Greater(), cmp_left, loma_ir.ConstInt(0))


            new_body = []
            for stmt in node.body:
                if isinstance(stmt, loma_ir.While):
                    next_ctr_ptr = loma_ir.Var('ctr' + str(self.ctr_idx + 1) + '_ptr', t = loma_ir.Int())
                    new_body.append(loma_ir.Assign(next_ctr_ptr,
                        loma_ir.BinaryOp(loma_ir.Sub(), next_ctr_ptr, loma_ir.ConstInt(1))))
                new_body.append(self.mutate_stmt(stmt))
            
            if cur_ctr_idx == 0:
                new_body.append(loma_ir.Assign(
                    loma_ir.Var(cur_ctr_name), loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.Var(cur_ctr_name), loma_ir.ConstInt(1))
                ))
            else:
                new_body.append(loma_ir.Assign(
                    loma_ir.ArrayAccess(loma_ir.Var(cur_ctr_name), loma_ir.Var(cur_ctr_name + '_ptr')),
                    loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.ArrayAccess(loma_ir.Var(cur_ctr_name), loma_ir.Var(cur_ctr_name + '_ptr')), loma_ir.ConstInt(1))
                ))
            new_body = irmutator.flatten(new_body)
            return loma_ir.While(new_cond, node.max_iter, new_body, lineno = node.lineno)

        def mutate_var(self, node):
            if self.in_assign:
                target = f'_adj_{str(self.adj_count)}'
                self.adj_count += 1
                self.adj_declaration.append(loma_ir.Declare(target, t=node.t))
                target_expr = loma_ir.Var(target, t=node.t)
                self.adj_accum_stmts += \
                    accum_deriv(var_to_differential(node, self.var_to_dvar),
                        target_expr, overwrite = False, simd = self.simd)
                return [accum_deriv(target_expr, self.adj, overwrite = True, simd = self.simd)]
            else:
                return [accum_deriv(var_to_differential(node, self.var_to_dvar),
                    self.adj, overwrite = False, simd = self.simd)]

        def mutate_const_float(self, node):
            return []

        def mutate_const_int(self, node):
            return []

        def mutate_array_access(self, node):
            if self.in_assign:
                target = f'_adj_{str(self.adj_count)}'
                self.adj_count += 1
                self.adj_declaration.append(loma_ir.Declare(target, t=node.t))
                target_expr = loma_ir.Var(target, t=node.t)
                self.adj_accum_stmts += \
                    accum_deriv(var_to_differential(node, self.var_to_dvar),
                        target_expr, overwrite = False)
                return [accum_deriv(target_expr, self.adj, overwrite = True, simd = self.simd)]
            else:
                return [accum_deriv(var_to_differential(node, self.var_to_dvar),
                    self.adj, overwrite = False, simd = self.simd)]

        def mutate_struct_access(self, node):
            if self.in_assign:
                target = f'_adj_{str(self.adj_count)}'
                self.adj_count += 1
                self.adj_declaration.append(loma_ir.Declare(target, t=node.t))
                target_expr = loma_ir.Var(target, t=node.t)
                self.adj_accum_stmts += \
                    accum_deriv(var_to_differential(node, self.var_to_dvar),
                        target_expr, overwrite = False, simd = self.simd)
                return [accum_deriv(target_expr, self.adj, overwrite = True, simd = self.simd)]
            else:
                return [accum_deriv(var_to_differential(node, self.var_to_dvar),
                    self.adj, overwrite = False, simd = self.simd)]

        def mutate_add(self, node):
            left = self.mutate_expr(node.left)
            right = self.mutate_expr(node.right)
            return left + right

        def mutate_sub(self, node):
            old_adj = self.adj
            left = self.mutate_expr(node.left)
            self.adj = loma_ir.BinaryOp(loma_ir.Sub(),
                loma_ir.ConstFloat(0.0), old_adj)
            right = self.mutate_expr(node.right)
            self.adj = old_adj
            return left + right

        def mutate_mul(self, node):
            # z = x * y
            # dz/dx = dz * y
            # dz/dy = dz * x
            old_adj = self.adj
            self.adj = loma_ir.BinaryOp(loma_ir.Mul(),
                node.right, old_adj)
            left = self.mutate_expr(node.left)
            self.adj = loma_ir.BinaryOp(loma_ir.Mul(),
                node.left, old_adj)
            right = self.mutate_expr(node.right)
            self.adj = old_adj
            return left + right

        def mutate_div(self, node):
            # z = x / y
            # dz/dx = dz / y
            # dz/dy = - dz * x / y^2
            old_adj = self.adj
            self.adj = loma_ir.BinaryOp(loma_ir.Div(),
                old_adj, node.right)
            left = self.mutate_expr(node.left)
            # - dz
            self.adj = loma_ir.BinaryOp(loma_ir.Sub(),
                loma_ir.ConstFloat(0.0), old_adj)
            # - dz * x
            self.adj = loma_ir.BinaryOp(loma_ir.Mul(),
                self.adj, node.left)
            # - dz * x / y^2
            self.adj = loma_ir.BinaryOp(loma_ir.Div(),
                self.adj, loma_ir.BinaryOp(loma_ir.Mul(), node.right, node.right))
            right = self.mutate_expr(node.right)
            self.adj = old_adj
            return left + right

        def mutate_call(self, node):
            match node.id:
                case 'sin':
                    assert len(node.args) == 1
                    old_adj = self.adj
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        loma_ir.Call(\
                            'cos',
                            node.args,
                            lineno = node.lineno,
                            t = node.t),
                        old_adj,
                        lineno = node.lineno)
                    ret = self.mutate_expr(node.args[0]) 
                    self.adj = old_adj
                    return ret
                case 'cos':
                    assert len(node.args) == 1
                    old_adj = self.adj
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Sub(),
                        loma_ir.ConstFloat(0.0),
                        loma_ir.BinaryOp(
                            loma_ir.Mul(),
                            loma_ir.Call(\
                                'sin',
                                node.args,
                                lineno = node.lineno,
                                t = node.t),
                            self.adj,
                            lineno = node.lineno),
                        lineno = node.lineno)
                    ret = self.mutate_expr(node.args[0]) 
                    self.adj = old_adj
                    return ret
                case 'sqrt':
                    assert len(node.args) == 1
                    # y = sqrt(x)
                    # dx = (1/2) * dy / y
                    old_adj = self.adj
                    sqrt = loma_ir.Call(\
                        'sqrt',
                        node.args,
                        lineno = node.lineno,
                        t = node.t)
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        loma_ir.ConstFloat(0.5), self.adj,
                        lineno = node.lineno)
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Div(),
                        self.adj, sqrt,
                        lineno = node.lineno)
                    ret = self.mutate_expr(node.args[0])
                    self.adj = old_adj
                    return ret
                case 'pow':
                    assert len(node.args) == 2
                    # y = pow(x0, x1)
                    # dx0 = dy * x1 * pow(x0, x1 - 1)
                    # dx1 = dy * pow(x0, x1) * log(x0)
                    base_expr = node.args[0]
                    exp_expr = node.args[1]

                    old_adj = self.adj
                    # base term
                    self.adj = loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        self.adj, exp_expr,
                        lineno = node.lineno)
                    exp_minus_1 = loma_ir.BinaryOp(\
                        loma_ir.Sub(),
                        exp_expr, loma_ir.ConstFloat(1.0),
                        lineno = node.lineno)
                    pow_exp_minus_1 = loma_ir.Call(\
                        'pow',
                        [base_expr, exp_minus_1],
                        lineno = node.lineno,
                        t = node.t)
                    self.adj = loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        self.adj, pow_exp_minus_1,
                        lineno = node.lineno)
                    base_stmts = self.mutate_expr(base_expr)
                    self.adj = old_adj

                    # exp term
                    pow_expr = loma_ir.Call(\
                        'pow',
                        [base_expr, exp_expr],
                        lineno = node.lineno,
                        t = node.t)
                    self.adj = loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        self.adj, pow_expr,
                        lineno = node.lineno)
                    log = loma_ir.Call(\
                        'log',
                        [base_expr],
                        lineno = node.lineno)
                    self.adj = loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        self.adj, log,
                        lineno = node.lineno)
                    exp_stmts = self.mutate_expr(exp_expr)
                    self.adj = old_adj
                    return base_stmts + exp_stmts
                case 'exp':
                    assert len(node.args) == 1
                    exp = loma_ir.Call(\
                        'exp',
                        node.args,
                        lineno = node.lineno,
                        t = node.t)
                    old_adj = self.adj
                    self.adj = loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        self.adj, exp,
                        lineno = node.lineno)
                    ret = self.mutate_expr(node.args[0])
                    self.adj = old_adj
                    return ret
                case 'log':
                    assert len(node.args) == 1
                    old_adj = self.adj
                    self.adj = loma_ir.BinaryOp(\
                        loma_ir.Div(),
                        self.adj, node.args[0])
                    ret = self.mutate_expr(node.args[0])
                    self.adj = old_adj
                    return ret
                case 'int2float':
                    # don't propagate the derivatives
                    return []
                case 'float2int':
                    # don't propagate the derivatives
                    return []
                case _:
                    original_func = funcs[node.id]
                    new_args = []
                    idx = 0
                    for arg in node.args:
                        if isinstance(original_func.args[idx].i, loma_ir.In):
                            new_args.append(arg)
                        new_args.append(var_to_differential(arg, self.var_to_dvar))
                        idx += 1
                    if original_func.ret_type is not None:
                        new_args.append(self.adj)
                    return [loma_ir.CallStmt(loma_ir.Call(\
                        func_to_rev[node.id],
                        new_args))]

    return RevDiffMutator().mutate_function_def(func)