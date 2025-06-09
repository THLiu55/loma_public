import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff

def Log(title: str, msg: str):
    print("***********************")
    print(title)
    print(msg)
    print("***********************")

def forward_diff(diff_func_id : str,
                 structs : dict[str, loma_ir.Struct],
                 funcs : dict[str, loma_ir.func],
                 diff_structs : dict[str, loma_ir.Struct],
                 func : loma_ir.FunctionDef,
                 func_to_fwd : dict[str, str]) -> loma_ir.FunctionDef:
    """ Given a primal loma function func, apply forward differentiation
        and return a function that computes the total derivative of func.

        For example, given the following function:
        def square(x : In[float]) -> float:
            return x * x
        and let diff_func_id = 'd_square', forward_diff() should return
        def d_square(x : In[_dfloat]) -> _dfloat:
            return make__dfloat(x.val * x.val, x.val * x.dval + x.dval * x.val)
        where the class _dfloat is
        class _dfloat:
            val : float
            dval : float
        and the function make__dfloat is
        def make__dfloat(val : In[float], dval : In[float]) -> _dfloat:
            ret : _dfloat
            ret.val = val
            ret.dval = dval
            return ret

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
        func_to_fwd - mapping from primal function ID to its forward differentiation
    """

    # HW1 happens here. Modify the following IR mutators to perform
    # forward differentiation.

    # Apply the differentiation.
    class FwdDiffMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            self.ret_type = node.ret_type
            new_args = [\
                loma_ir.Arg(arg.id, autodiff.type_to_diff_type(diff_structs, arg.t), arg.i) \
                for arg in node.args]
            new_body = [self.mutate_stmt(stmt) for stmt in node.body]
            new_body = irmutator.flatten(new_body)
            return loma_ir.FunctionDef(\
                diff_func_id,
                new_args,
                new_body,
                node.is_simd,
                autodiff.type_to_diff_type(diff_structs, node.ret_type),
                lineno = node.lineno)

        def mutate_return(self, node):
            stmts = []
            # loma does not support returning multiple values,
            # so to return the primal + differentials, we need
            # to do some massage
            mutated_val = self.mutate_expr(node.val)
            diff_type = autodiff.type_to_diff_type(diff_structs, self.ret_type)
            # It's easier to divide this into two cases based
            # on the return type: Float(), and others
            if isinstance(self.ret_type, loma_ir.Float):
                val, dval = mutated_val
                call_expr = loma_ir.Call('make__dfloat',
                    [val, dval], lineno = node.lineno)
                return loma_ir.Return(call_expr)
            else:
                val, _ = mutated_val
                return loma_ir.Return(val)

        def mutate_declare(self, node):
            # Turn the declaration into a struct
            # declaration, followed by assignments
            diff_type = autodiff.type_to_diff_type(diff_structs, node.t)
            if node.val is not None:
                if isinstance(node.t, loma_ir.Float):
                    val, dval = self.mutate_expr(node.val)
                    call_expr = loma_ir.Call('make__dfloat',
                        [val, dval], lineno = node.lineno)
                    return loma_ir.Declare(\
                        node.target,
                        diff_type,
                        val = call_expr,
                        lineno = node.lineno)
                else:
                    val, _ = self.mutate_expr(node.val)
                    return loma_ir.Declare(\
                        node.target,
                        diff_type,
                        val = val,
                        lineno = node.lineno)
            else:
                return loma_ir.Declare(\
                    node.target,
                    diff_type,
                    lineno = node.lineno)

        def mutate_assign(self, node):
            if isinstance(node.val.t, loma_ir.Float):
                val, dval = self.mutate_expr(node.val)
                call_expr = loma_ir.Call('make__dfloat',
                    [val, dval], lineno = node.lineno)
                tgt_val, _ = self.mutate_expr(node.target)
                assert isinstance(tgt_val, loma_ir.StructAccess)
                return loma_ir.Assign(tgt_val.struct, call_expr)
            else:
                val, _ = self.mutate_expr(node.val)
                tgt_val, _ = self.mutate_expr(node.target)
                return loma_ir.Assign(tgt_val, val)
        
        def mutate_ifelse(self, node):
            cond = node.cond
            cond = loma_ir.BinaryOp(cond.op, loma_ir.StructAccess(loma_ir.Var(cond.left.id), 'val'), cond.right)
            then_stmts = [self.mutate_stmt(stmt) for stmt in node.then_stmts]
            then_stmts = irmutator.flatten(then_stmts)
            else_stmts = [self.mutate_stmt(stmt) for stmt in node.else_stmts]
            else_stmts = irmutator.flatten(else_stmts)
            return loma_ir.IfElse(cond, then_stmts, else_stmts, lineno=node.lineno)

        def mutate_while(self, node):
            return loma_ir.While(node.cond, node.max_iter,
                [self.mutate_stmt(stmt) for stmt in node.body],
                lineno=node.lineno)

        def mutate_const_float(self, node):
            return node, loma_ir.ConstFloat(0.0)

        def mutate_const_int(self, node):
            return node, None

        def mutate_var(self, node):
            if isinstance(node.t, loma_ir.Float):
                return (loma_ir.StructAccess(loma_ir.Var(node.id), 'val'),
                    loma_ir.StructAccess(loma_ir.Var(node.id), 'dval'))
            else:
                return node, None

        def mutate_array_access(self, node):
            array, _ = self.mutate_expr(node.array)
            index, _ = self.mutate_expr(node.index)
            if isinstance(node.t, loma_ir.Float):
                val = loma_ir.StructAccess(loma_ir.ArrayAccess(\
                    array, index), 'val',
                    lineno = node.lineno,
                    t = node.t)
                dval = loma_ir.StructAccess(loma_ir.ArrayAccess(\
                    array, index), 'dval',
                    lineno = node.lineno,
                    t = node.t)
                return val, dval
            else:
                return loma_ir.ArrayAccess(\
                    array, index, lineno = node.lineno, t = node.t), None

        def mutate_struct_access(self, node):
            struct, _ = self.mutate_expr(node.struct)
            if isinstance(node.t, loma_ir.Float):
                val = loma_ir.StructAccess(loma_ir.StructAccess(\
                    struct, node.member_id), 'val',
                    lineno = node.lineno,
                    t = node.t)
                dval = loma_ir.StructAccess(loma_ir.StructAccess(\
                    struct, node.member_id), 'dval',
                    lineno = node.lineno,
                    t = node.t)
                return val, dval
            else:
                return loma_ir.StructAccess(\
                    struct, node.member_id, lineno = node.lineno, t = node.t), None

        def mutate_add(self, node):
            l_val, l_dval = self.mutate_expr(node.left)
            r_val, r_dval = self.mutate_expr(node.right)
            val = loma_ir.BinaryOp(\
                loma_ir.Add(),
                l_val,
                r_val,
                lineno = node.lineno,
                t = node.t)
            if l_dval is not None:
                assert r_dval is not None
                dval = loma_ir.BinaryOp(\
                    loma_ir.Add(),
                    l_dval,
                    r_dval,
                    lineno = node.lineno,
                    t = node.t)
            else:
                dval = None
            return val, dval

        def mutate_sub(self, node):
            l_val, l_dval = self.mutate_expr(node.left)
            r_val, r_dval = self.mutate_expr(node.right)
            val = loma_ir.BinaryOp(\
                loma_ir.Sub(),
                l_val,
                r_val,
                lineno = node.lineno,
                t = node.t)
            if l_dval is not None:
                assert r_dval is not None
                dval = loma_ir.BinaryOp(\
                    loma_ir.Sub(),
                    l_dval,
                    r_dval,
                    lineno = node.lineno,
                    t = node.t)
            else:
                dval = None
            return val, dval

        def mutate_mul(self, node):
            l_val, l_dval = self.mutate_expr(node.left)
            r_val, r_dval = self.mutate_expr(node.right)
            val = loma_ir.BinaryOp(\
                loma_ir.Mul(),
                l_val,
                r_val,
                lineno = node.lineno,
                t = node.t)
            if l_dval is not None:
                l_ = loma_ir.BinaryOp(\
                    loma_ir.Mul(),
                    l_dval,
                    r_val,
                    lineno = node.lineno
                )
                r_ = loma_ir.BinaryOp(\
                    loma_ir.Mul(),
                    l_val,
                    r_dval,
                    lineno = node.lineno
                )
                dval = loma_ir.BinaryOp(\
                    loma_ir.Add(),
                    l_,
                    r_,
                    lineno = node.lineno,
                    t = node.t)
            else:
                dval = None
            return val, dval

        def mutate_div(self, node):
            l_val, l_dval = self.mutate_expr(node.left)
            r_val, r_dval = self.mutate_expr(node.right)
            val = loma_ir.BinaryOp(\
                loma_ir.Div(),
                l_val,
                r_val,
                lineno = node.lineno,
                t = node.t)
            if l_dval is not None:
                denom = loma_ir.BinaryOp(\
                    loma_ir.Mul(),
                    r_val,
                    r_val
                )
                dl_pr = loma_ir.BinaryOp(\
                    loma_ir.Mul(),
                    l_dval,
                    r_val,
                    lineno = node.lineno
                )
                pl_dr = loma_ir.BinaryOp(\
                    loma_ir.Mul(),
                    l_val,
                    r_dval,
                    lineno = node.lineno
                )
                numerator = loma_ir.BinaryOp(\
                    loma_ir.Sub(),
                    dl_pr,
                    pl_dr,
                    lineno = node.lineno
                )
                dval = loma_ir.BinaryOp(\
                    loma_ir.Div(),
                    numerator,
                    denom,
                    lineno = node.lineno,
                    t = node.t)
            return val, dval

        def mutate_call(self, node):
            new_args = [self.mutate_expr(arg) for arg in node.args]
            match node.id:
                case 'sin':
                    assert len(new_args) == 1
                    val, dval = new_args[0]
                    sin = loma_ir.Call('sin', [val],
                        lineno = node.lineno, t = node.t)
                    cos = loma_ir.Call('cos', [val],
                        lineno = node.lineno, t = node.t)
                    return sin, loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        cos,
                        dval,
                        lineno = node.lineno,
                        t = node.t)
                case 'cos':
                    assert len(new_args) == 1
                    val, dval = new_args[0]
                    sin = loma_ir.Call(\
                        'sin',
                        [val],
                        lineno = node.lineno,
                        t = node.t)
                    cos = loma_ir.Call(\
                        'cos',
                        [val],
                        lineno = node.lineno,
                        t = node.t)
                    neg_sin = loma_ir.BinaryOp(\
                        loma_ir.Sub(),
                        loma_ir.ConstFloat(0.0),
                        sin,
                        lineno = node.lineno,
                        t = node.t)
                    return cos, loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        neg_sin,
                        dval,
                        lineno = node.lineno,
                        t = node.t)
                case 'sqrt':
                    assert len(new_args) == 1
                    val, dval = new_args[0]
                    sqrt = loma_ir.Call(\
                        'sqrt',
                        [val],
                        lineno = node.lineno,
                        t = node.t)
                    two_sqrt = loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        loma_ir.ConstFloat(2.0), sqrt,
                        lineno = node.lineno,
                        t = node.t)
                    return sqrt, loma_ir.BinaryOp(\
                        loma_ir.Div(),
                        dval,
                        two_sqrt,
                        lineno = node.lineno,
                        t = node.t)
                case 'pow':
                    assert len(new_args) == 2
                    base_val, base_dval = new_args[0]
                    exp_val, exp_dval = new_args[1]
                    exp_minus_1 = loma_ir.BinaryOp(\
                        loma_ir.Sub(),
                        exp_val, loma_ir.ConstFloat(1.0),
                        lineno = node.lineno,
                        t = node.t)
                    pow_exp_minus_1 = loma_ir.Call(\
                        'pow',
                        [base_val, exp_minus_1],
                        lineno = node.lineno,
                        t = node.t)
                    exp_pow_exp_minus_1 = loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        exp_val, pow_exp_minus_1,
                        lineno = node.lineno,
                        t = node.t)
                    base_dval_term = loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        base_dval, exp_pow_exp_minus_1,
                        lineno = node.lineno,
                        t = node.t)
                    pow = loma_ir.Call(\
                        'pow',
                        [base_val, exp_val],
                        lineno = node.lineno,
                        t = node.t)
                    log = loma_ir.Call(\
                        'log',
                        [base_val],
                        lineno = node.lineno,
                        t = node.t)
                    pow_log = loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        pow, log,
                        lineno = node.lineno,
                        t = node.t)
                    exp_dval_term = loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        exp_dval, pow_log,
                        lineno = node.lineno,
                        t = node.t)
                    return pow, loma_ir.BinaryOp(\
                        loma_ir.Add(),
                        base_dval_term, exp_dval_term,
                        lineno = node.lineno,
                        t = node.t)
                case 'exp':
                    assert len(new_args) == 1
                    val, dval = new_args[0]
                    exp = loma_ir.Call(\
                        'exp',
                        [val],
                        lineno = node.lineno,
                        t = node.t)
                    return exp, loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        dval,
                        exp,
                        lineno = node.lineno,
                        t = node.t)
                case 'log':
                    assert len(new_args) == 1
                    val, dval = new_args[0]
                    log = loma_ir.Call(\
                        'log',
                        [val],
                        lineno = node.lineno,
                        t = node.t)
                    return log, loma_ir.BinaryOp(\
                        loma_ir.Div(),
                        dval,
                        val,
                        lineno = node.lineno,
                        t = node.t)
                case 'int2float':
                    assert len(new_args) == 1
                    val, _ = new_args[0]
                    ret = loma_ir.Call(\
                        'int2float',
                        [val],
                        lineno = node.lineno,
                        t = node.t)
                    return ret, loma_ir.ConstFloat(0.0)
                case 'float2int':
                    assert len(new_args) == 1
                    val, dval = new_args[0]
                    ret = loma_ir.Call(\
                        'float2int',
                        [val],
                        lineno = node.lineno,
                        t = node.t)
                    return ret, None
                case 'mpi_rank':
                    print("Inside mutate_call: mpi_rank")
                    print(node)
                    return node
                case 'mpi_size':
                    return node
                case 'init_mpi_env':
                    return node
                case 'mpi_chunk_size':
                    assert len(new_args) == 0
                    return loma_ir.Call('mpi_chunk_size',
                        node.args, lineno = node.lineno), 0
                case 'scatter':
                    print(f"Length of scatter functions' args  {len(new_args)}")
                    print(f"Scatter args: {new_args}")
                    print(f"Scatter node args: {node.args}")
                    return loma_ir.Call('scatter',
                        [new_args[0][0], new_args[1][0]],
                        lineno = node.lineno)
                case 'gather':
                    return loma_ir.Call('gather',
                        [new_args[0][0], new_args[1][0]],
                        lineno = node.lineno)
                case default:
                    final_args = []
                    original_func = funcs[node.id]
                    idx = 0
                    for val, dval in new_args:
                        tmp_args = [val, dval]
                        if isinstance(original_func.args[idx].i, loma_ir.In):
                            expr = loma_ir.Call('make__dfloat', tmp_args)
                        else:
                            expr = node.args[idx]
                        final_args.append(expr)
                        idx += 1
                    expr = loma_ir.Call('_d_fwd_' + node.id, irmutator.flatten(final_args))
                    if original_func.ret_type is None:
                        return expr
                    return loma_ir.StructAccess(expr, 'val'), \
                        loma_ir.StructAccess(expr, 'dval')
                
    

    return FwdDiffMutator().mutate_function_def(func)