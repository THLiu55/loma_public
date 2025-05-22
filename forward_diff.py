import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff

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
            # HW1: TODO
            new_args = [ loma_ir.Arg( arg.id, 
                                     autodiff.type_to_diff_type( diff_structs, arg.t), arg.i ) for arg in node.args ]
            new_node = loma_ir.FunctionDef(\
                diff_func_id, 
                new_args, 
                [self.mutate_stmt(stmt) for stmt in node.body], 
                node.is_simd, 
                autodiff.type_to_diff_type(diff_structs, node.ret_type), 
                node.lineno )
            return new_node

        def mutate_return(self, node):
            # HW1: TODO
            val, dval = self.mutate_expr(node.val)

            if node.val.t == loma_ir.Float():
                new_node = loma_ir.Call( 
                    "make__dfloat",
                    [val, dval],
                    lineno = node.lineno
                )
            else:
                new_node = val
            return loma_ir.Return(
                new_node,
                node.lineno
            )

        def mutate_declare(self, node):
            # HW1: TODO
            new_node = None
            if node.val is not None:
                if node.val.t == loma_ir.Int() or isinstance(node.val.t, loma_ir.Struct):
                    new_node = node.val
                else:
                    new_node_val, new_node_dval = self.mutate_expr(node.val)
                    new_node = loma_ir.Call(
                        "make__dfloat",
                        [new_node_val, new_node_dval],
                        lineno=node.lineno
                    )
                
            return loma_ir.Declare(
                node.target,
                autodiff.type_to_diff_type(diff_structs, node.t),
                new_node,
                lineno=node.lineno
            )

        def mutate_assign(self, node):
            # HW1: TODO
            new_target = node.target
            # We shuold mutate if we have array acess (only mutate the indexing inside it!)
            if isinstance(node.target, loma_ir.ArrayAccess):
                t, dt = self.mutate_expr(node.target.index)
                new_target = loma_ir.ArrayAccess(node.target.array, t, node.target.lineno, node.target.t)
            val, dval = self.mutate_expr(node.val)
            if node.target.t == loma_ir.Int() or isinstance(node.target.t, loma_ir.Struct):
                return loma_ir.Assign(new_target, val)
            else:
                return loma_ir.Assign(
                    new_target,
                    loma_ir.Call( "make__dfloat",[ val, dval ])
                )

        def mutate_ifelse(self, node):
            # HW3: TODO
            new_cond, _ = self.mutate_expr(node.cond)
            new_then_stmts = [self.mutate_stmt(stmt) for stmt in node.then_stmts]
            new_else_stmts = [self.mutate_stmt(stmt) for stmt in node.else_stmts]
            # Important: mutate_stmt can return a list of statements. We need to flatten the lists.
            new_then_stmts = irmutator.flatten(new_then_stmts)
            new_else_stmts = irmutator.flatten(new_else_stmts)
            return loma_ir.IfElse(\
                new_cond,
                new_then_stmts,
                new_else_stmts,
                lineno = node.lineno)


        def mutate_while(self, node):
            # HW3: TODO
            # new_cond = self.mutate_expr(node.cond)
            new_body = [self.mutate_stmt(stmt) for stmt in node.body]
            # Important: mutate_stmt can return a list of statements. We need to flatten the list.
            new_body =  irmutator.flatten(new_body)
            return loma_ir.While(\
                node.cond,
                node.max_iter,
                new_body,
            lineno = node.lineno)

            return super().mutate_while(node)

        def mutate_const_float(self, node):
            # HW1: TODO
            return node, loma_ir.ConstFloat(0.0)

        def mutate_const_int(self, node):
            # HW1: TODO
            return node, loma_ir.ConstFloat(0.0)

        def mutate_var(self, node):
            # HW1: TODO
            # Struc and node should be the same
            if node.t == loma_ir.Int() or isinstance( node.t, loma_ir.Struct):
                return node, loma_ir.ConstFloat(0.0)
            else:
                val = loma_ir.StructAccess(node, "val", lineno=node.lineno, t=node.t)
                dval = loma_ir.StructAccess(node, "dval", lineno=node.lineno, t=node.t)
                return val, dval
            # return super().mutate_var(node)

        def mutate_array_access(self, node):
            # HW1: TODO
            new_index_val, _ = self.mutate_expr(node.index)
            if new_index_val.t != loma_ir.Int():
                raise TypeError(f"Cannot use '{new_index_val.t}' as a array index; ")
            new_array, _ = self.mutate_expr(node.array)

            new_node =  loma_ir.ArrayAccess(\
                node.array,
                new_index_val,
                lineno = node.lineno,
                t = node.t
            )
            if node.t == loma_ir.Int():
                return new_node, loma_ir.ConstFloat(0.0)
            else:
                val = loma_ir.StructAccess(new_node, "val", lineno=node.lineno, t=node.t)
                dval = loma_ir.StructAccess(new_node, "dval", lineno=node.lineno, t=node.t)
            return val, dval

        def mutate_struct_access(self, node):
            # HW1: TODO
            # Need to access val and dval for _dfloat, otherwise we simply return itself and a 0 dval for consistency
            if node.t == loma_ir.Float():
                val = loma_ir.StructAccess(\
                    node,
                    "val",
                    lineno = node.lineno,
                    t = node.t)
                dval = loma_ir.StructAccess(\
                    node,
                    "dval",
                    lineno = node.lineno,
                    t = node.t)
            else:
                val = node
                dval = loma_ir.ConstFloat(0.0)

            return val, dval

        def mutate_add(self, node):
            # HW1: TODO
            left_val, left_dval = self.mutate_expr(node.left)
            right_val, right_dval = self.mutate_expr(node.right)

            val = loma_ir.BinaryOp(
                loma_ir.Add(),
                left_val,
                right_val,
                lineno=node.lineno,
                t=node.t
            )

            dval = loma_ir.BinaryOp(
                loma_ir.Add(),
                left_dval,
                right_dval,
                lineno=node.lineno,
                t=node.t
            )
            return val, dval

        def mutate_sub(self, node):
            left_val, left_dval = self.mutate_expr(node.left)
            right_val, right_dval = self.mutate_expr(node.right)

            val = loma_ir.BinaryOp( loma_ir.Sub(), left_val, right_val, t=node.t )
            dval = loma_ir.BinaryOp( loma_ir.Sub(), left_dval, right_dval, t=loma_ir.Float() )
            return val, dval


        def mutate_mul(self, node):
            # HW1: TODO
            left_val, left_dval = self.mutate_expr(node.left)
            right_val, right_dval = self.mutate_expr(node.right)

            res_left = loma_ir.BinaryOp( loma_ir.Mul(), left_dval, right_val )
            res_right = loma_ir.BinaryOp( loma_ir.Mul(), left_val, right_dval )

            val = loma_ir.BinaryOp( loma_ir.Mul(), left_val, right_val, t=node.t )
            dval = loma_ir.BinaryOp( loma_ir.Add(), res_left, res_right, t=loma_ir.Float() )
            return val, dval

        def mutate_div(self, node):
            # HW1: TODO
            left_val, left_dval = self.mutate_expr(node.left)
            right_val, right_dval = self.mutate_expr(node.right)
            
            res_left = loma_ir.BinaryOp( loma_ir.Mul(), left_dval, right_val )
            res_right = loma_ir.BinaryOp( loma_ir.Mul(), left_val, right_dval )

            res_molecular = loma_ir.BinaryOp( loma_ir.Sub(), res_left, res_right )
            res_denominor = loma_ir.BinaryOp( loma_ir.Mul(), right_val, right_val )

            val = loma_ir.BinaryOp( loma_ir.Div(), left_val, right_val , t=node.t)
            dval = loma_ir.BinaryOp( loma_ir.Div(), res_molecular, res_denominor, t=loma_ir.Float() )
            return  val, dval 

        def mutate_call(self, node):
            # HW1: TODO
            if node.id == 'sin':
                assert(len(node.args) == 1)
                arg_val, arg_dval  = self.mutate_expr( node.args[0] )
                val = loma_ir.Call(
                    node.id,
                    [arg_val],
                    lineno=node.lineno,
                    t=node.t)
                cos_val = loma_ir.Call(
                    'cos',
                    [arg_val],
                    lineno = node.lineno,
                    t = node.t)
                dval = loma_ir.BinaryOp( 
                    loma_ir.Mul(),
                    arg_dval,
                    cos_val,
                    lineno=node.lineno,
                    t=loma_ir.Float())
            elif node.id == 'cos':
                assert(len(node.args) == 1)
                arg_val, arg_dval  = self.mutate_expr( node.args[0] )
                val = loma_ir.Call(
                    node.id,
                    [arg_val],
                    lineno=node.lineno,
                    t=node.t)
                sin_val = loma_ir.Call(
                    'sin',
                    [arg_val],
                    lineno = node.lineno)
                df_val = loma_ir.BinaryOp( 
                    loma_ir.Mul(),
                    arg_dval,
                    sin_val,
                    lineno=node.lineno,
                    t=loma_ir.Float()
                )
                dval = loma_ir.BinaryOp( 
                    loma_ir.Sub(),
                    loma_ir.ConstFloat(0.0),
                    df_val,
                    lineno=node.lineno,
                    t=loma_ir.Float()
                )
            elif node.id == 'sqrt':
                assert(len(node.args) == 1)
                arg_val, arg_dval  = self.mutate_expr( node.args[0] )

                val = loma_ir.Call(
                    'sqrt',
                    [arg_val],
                    lineno=node.lineno,
                    t=node.t
                )

                denom = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    loma_ir.ConstFloat(2.0),
                    val,
                    lineno=node.lineno)
                dval = loma_ir.BinaryOp(
                    loma_ir.Div(),
                    arg_dval,
                    denom,
                    lineno=node.lineno,
                    t=loma_ir.Float()
                )
            elif node.id == 'pow':
                assert(len(node.args) == 2)
                x_val, x_dval = self.mutate_expr(node.args[0])
                y_val, y_dval = self.mutate_expr(node.args[1])
                
                # x^y
                val = loma_ir.Call(
                    'pow',
                    [x_val, y_val],
                    lineno=node.lineno,
                    t=node.t
                )

                # y * x^(y-1)
                y_minus_1 = loma_ir.BinaryOp(loma_ir.Sub(), y_val, loma_ir.ConstFloat(1.0))
                x_pow_y_minus_1 = loma_ir.Call('pow', [x_val, y_minus_1])
                dy_dx_term = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    x_dval,
                    loma_ir.BinaryOp(loma_ir.Mul(), y_val, x_pow_y_minus_1)
                )

                # x^y * log(x)
                log_x = loma_ir.Call('log', [x_val])
                dy_dy_term = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    y_dval,
                    loma_ir.BinaryOp(loma_ir.Mul(), val, log_x)
                )

                dval = loma_ir.BinaryOp(
                    loma_ir.Add(),
                    dy_dx_term,
                    dy_dy_term,
                    lineno=node.lineno,
                    t=loma_ir.Float()
                )
            elif node.id == 'exp':
                assert(len(node.args) == 1)
                arg_val, arg_dval  = self.mutate_expr( node.args[0] )

                val = loma_ir.Call(
                    'exp',
                    [arg_val],
                    lineno=node.lineno
                )

                dval = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    arg_dval,
                    val,
                    lineno=node.lineno
                )
            elif node.id == 'log':
                assert(len(node.args) == 1)
                arg_val, arg_dval  = self.mutate_expr( node.args[0] )
                val = loma_ir.Call(
                    'log',
                    [arg_val],
                    lineno=node.lineno
                )
                inv = loma_ir.BinaryOp(
                    loma_ir.Div(),
                    loma_ir.ConstFloat(1.0),
                    arg_val,
                    lineno=node.lineno
                    )
                dval = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    inv,
                    arg_dval,
                    lineno=node.lineno
                    )
            elif node.id == 'int2float': 
                assert(len(node.args) == 1)

                arg_val, _ = self.mutate_expr(node.args[0])

                val = loma_ir.Call(
                    'int2float',
                    [arg_val],
                    lineno=node.lineno,
                    t=loma_ir.Float()
                )

                dval = loma_ir.ConstFloat(0.0)
            elif node.id == 'float2int' :
                assert len(node.args) == 1
                arg_val, arg_dval  = self.mutate_expr( node.args[0] )
                val =  loma_ir.Call(
                    'float2int',
                    [arg_val],
                    lineno=node.lineno,
                    t = loma_ir.Int()
                )
                dval = loma_ir.ConstFloat(0.0)
                
            else:
                # print("Undefined differentiable interior function!")
                # assert False, f'Unhandled function {node.id}'
                new_id = func_to_fwd.get(node.id)
                new_args = []
                for arg in node.args:
                    arg_val, arg_dval = self.mutate_expr(arg)
                    if arg.t == loma_ir.Float() and not isinstance(arg, loma_ir.Var) :
                        new_args.append(loma_ir.Call(\
                            "make__dfloat", 
                            [arg_val, arg_dval],
                            t=autodiff.type_to_diff_type( diff_structs, arg.t)))
                    else:
                        new_args.append(arg)

                
                new_call = loma_ir.Call( id=new_id, args=new_args, t=node.t )
                if node.t == loma_ir.Float():
                    val = loma_ir.StructAccess( new_call,  "val", lineno=node.lineno, t=node.t)
                    dval = loma_ir.StructAccess( new_call, "dval", lineno=node.lineno, t=node.t)   
                else:
                    val = new_call
                    dval = loma_ir.ConstFloat(0.0)
            return val, dval

        def mutate_call_stmt(self, node):
            val, d_val = self.mutate_expr(node.call)
            return loma_ir.CallStmt( val, lineno = node.lineno)

        def mutate_less(self, node):
            l_val, l_dval = self.mutate_expr(node.left)
            r_val, r_dval = self.mutate_expr(node.right)
            return loma_ir.BinaryOp(\
                loma_ir.Less(),
                l_val,
                r_val,
                lineno = node.lineno,
                t = node.t), None

        def mutate_less_equal(self, node):
            l_val, l_dval = self.mutate_expr(node.left)
            r_val, r_dval = self.mutate_expr(node.right)
            return loma_ir.BinaryOp(\
                loma_ir.LessEqual(),
                l_val,
                r_val,
                lineno = node.lineno,
                t = node.t), None

        def mutate_greater(self, node):
            l_val, l_dval = self.mutate_expr(node.left)
            r_val, r_dval = self.mutate_expr(node.right)
            return loma_ir.BinaryOp(\
                loma_ir.Greater(),
                l_val,
                r_val,
                lineno = node.lineno,
                t = node.t), None

        def mutate_greater_equal(self, node):
            l_val, l_dval = self.mutate_expr(node.left)
            r_val, r_dval = self.mutate_expr(node.right)
            return loma_ir.BinaryOp(\
                loma_ir.GreaterEqual(),
                l_val,
                r_val,
                lineno = node.lineno,
                t = node.t), None

        def mutate_equal(self, node):
            l_val, l_dval = self.mutate_expr(node.left)
            r_val, r_dval = self.mutate_expr(node.right)
            return loma_ir.BinaryOp(\
                loma_ir.Equal(),
                l_val,
                r_val,
                lineno = node.lineno,
                t = node.t), None

        def mutate_and(self, node):
            l_val, l_dval = self.mutate_expr(node.left)
            r_val, r_dval = self.mutate_expr(node.right)
            return loma_ir.BinaryOp(\
                loma_ir.And(),
                l_val,
                r_val,
                lineno = node.lineno,
                t = node.t), None

    return FwdDiffMutator().mutate_function_def(func)
