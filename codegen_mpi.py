# codegen_mpi.py
import codegen_c
import _asdl.loma as loma_ir
import compiler
from reverse_diff import random_id_generator

# ------------  Code-gen Visitor  ------------
class OpenMPICodegenVisitor(codegen_c.CCodegenVisitor):
    """
    负责把 Loma IR 中的函数转换成 OpenMPI 风格的并行 C 代码。
    基本思路：
        * 所有函数仍然是普通 C 函数（无 __kernel / __global）
        * 在线程 ID 处插入 MPI 的 rank
        * 在函数开头插入 rank/size 变量与 MPI_Comm_rank/size
        * atomic_add ➜ 简单 +=（如需严格原子性可再插入 MPI_Reduce）
    """

    def __init__(self, func_defs):
        super().__init__(func_defs)
        self.rank_var = "rank"
        self.size_var = "size"

    # ---------- function ----------
    def visit_function_def(self, node):
        # 函数签名
        self.total_size_var = '_mpi_total_size_' + random_id_generator(1)
        self.code += f'{codegen_c.type_to_string(node.ret_type)} {node.id}('
        for i, arg in enumerate(node.args):
            if i: self.code += ', '
            self.code += f'{codegen_c.type_to_string(arg)} {arg.id}'
        if len(node.args) > 0:
            self.code += ', '
        self.code += f'int {self.total_size_var}'
        self.code += ') {\n'


        # 记录输出参数
        self.byref_args = {arg.id for arg in node.args
                           if arg.i == loma_ir.Out() and not isinstance(arg.t, loma_ir.Array)}
        self.output_args = {arg.id for arg in node.args if arg.i == loma_ir.Out()}

        self.tab_count += 1

        # self.emit_tabs()
        # self.code +=  "MPI_Init(NULL, NULL);\n"

        # 插入 rank / size

        # 处理函数体
        for stmt in node.body:
            self.visit_stmt(stmt)

        # self.emit_tabs()
        # self.code +=  "MPI_Finalize();\n"

        self.tab_count -= 1
        self.emit_tabs(); self.code += '}\n'

    # ---------- helper ----------
    def is_output_arg(self, node):
        match node:
            case loma_ir.Var():           return node.id in self.output_args
            case loma_ir.ArrayAccess():   return self.is_output_arg(node.array)
            case loma_ir.StructAccess():  return self.is_output_arg(node.struct)
        return False

    # ---------- expression ----------
    def visit_expr(self, node):
        if isinstance(node, loma_ir.Call) and node.id == 'init_mpi_env':
            self.sendcounts_var = 'sendcounts_' + random_id_generator()
            self.displs_var = 'displs_' + random_id_generator()
            self.recv_counts_var = 'recvcounts_' + random_id_generator()
            self.mpi_base_var = 'mpi_base_' + random_id_generator()
            self.mpi_extra_var = 'mpi_extra_' + random_id_generator()

            code = f"\tint* {self.sendcounts_var} = NULL;\n"
            code += f"\tint* {self.displs_var} = NULL;\n"
            code += f"\t        int {self.mpi_base_var} = {self.total_size_var} / {node.args[1].id} ;\n"
            code += f"\t        int {self.mpi_extra_var} = {self.total_size_var} % {node.args[1].id};\n"
            code += f"\tint {self.recv_counts_var} =  {self.mpi_base_var} + ({node.args[0].id} < {self.mpi_extra_var}  ? 1 : 0);\n"
            code += f"\tif ({node.args[0].id} == 0) {{\n"
            code += f"\t        {self.sendcounts_var} = malloc({node.args[1].id} * sizeof(int));\n"
            code += f"\t         {self.displs_var}      = malloc({node.args[1].id} * sizeof(int));\n"
            code += "\t        int offset = 0;\n"
            code += f"\t        for (int i = 0; i < {node.args[1].id}; i++) {{\n"
            code += f"\t            {self.sendcounts_var}[i] = {self.mpi_base_var} + (i < {self.mpi_extra_var}  ? 1 : 0);\n"
            code += f"\t             {self.displs_var}[i] = offset;\n"
            code += f"\t            offset += {self.sendcounts_var}[i];\n"
            code += "\t        }\n"
            code += "\t    }"
            return code
        match node:
            case loma_ir.Call():
                if node.id == 'atomic_add':
                    if self.is_output_arg(node.args[0]):
                        a0 = self.visit_expr(node.args[0])
                        a1 = self.visit_expr(node.args[1])
                        return f'/* atomic_add naive */ ({a0} += {a1})'
                elif node.id == 'mpi_rank':
                    return f'MPI_Comm_rank(MPI_COMM_WORLD, &{node.args[0].id})'
                elif node.id == 'mpi_size':
                    return f'MPI_Comm_size(MPI_COMM_WORLD, &{node.args[0].id})'
                elif node.id == 'mpi_total_size':
                    return f'{self.total_size_var}'
                elif node.id == 'scatter':
                    # global_arr, local, total_size
                    src = self.visit_expr(node.args[0])   # global_arr
                    dst = self.visit_expr(node.args[1])   # local

                    # 生成：MPI_Scatterv(...)
                    return (
                        f'MPI_Scatterv('
                        f'{src}, {self.sendcounts_var}, {self.displs_var}, MPI_FLOAT, '
                        f'{dst}, {self.recv_counts_var}, MPI_FLOAT, '
                        f'0, MPI_COMM_WORLD)'
                    )
                elif node.id == 'gather':
                    # gather(local, global_arr, total_size)
                    src = self.visit_expr(node.args[0])   # local
                    dst = self.visit_expr(node.args[1])   # global_arr
                    return (
                        f'MPI_Gatherv('
                        f'{src}, {self.recv_counts_var}, MPI_FLOAT, '
                        f'{dst}, {self.sendcounts_var}, {self.displs_var}, MPI_FLOAT, '
                        f'0, MPI_COMM_WORLD)'
                    )
                elif node.id == 'mpi_chunk_size':
                    return f'({self.recv_counts_var})'


        return super().visit_expr(node)


# ------------  Top-level code-gen  ------------
def codegen_mpi(structs: dict[str, loma_ir.Struct],
                funcs  : dict[str, loma_ir.func]) -> str:
    """
    把结构体 + 函数 IR 转成 OpenMPI C 源码字符串
    """
    code = ''
    # 头文件
    code += '#include <mpi.h>\n#include <math.h>\n#include <stdlib.h>\n\n'

    ctype_structs = compiler.topo_sort_structs(structs)

    # -------- structs --------
    for s in ctype_structs:
        code += f'typedef struct {s.id} {{\n'
        for m in s.members:
            code += f'    {codegen_c.type_to_string(m.t)} {m.id};\n'
        code += f'}} {s.id};\n\n'

    # -------- forward decl --------
    for f in funcs.values():
        code += f'{codegen_c.type_to_string(f.ret_type)} {f.id}('
        for i, arg in enumerate(f.args):
            if i: code += ', '
            code += f'{codegen_c.type_to_string(arg)} {arg.id}'
        if f.is_simd:
            if len(f.args) > 0:
                code += ', '
            code += 'int total_work'
        code += ');\n'
    code += '\n'

    # -------- function bodies --------
    for f in funcs.values():
        vis = OpenMPICodegenVisitor(funcs)
        vis.visit_function(f)
        code += vis.code + '\n'

    return code
