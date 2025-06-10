from _asdl.loma import Float, Int, Array, Struct
import _asdl.loma as loma_ir

def get_flatten_info(loma_ty):
    """
    For any type `loma_ty` consisting only of float/int/fixed-length arrays/nested structs:
    - If it can be "flattened" into N floats (all fields are float or array<float>),
      return (N, "MPI_FLOAT", True)
    - If it can be "flattened" into N ints, return (N, "MPI_INT", True)
    - Otherwise, return (None, None, False)
    """

    if isinstance(loma_ty, Float):
        return 1, "MPI_FLOAT", True
    if isinstance(loma_ty, Int):
        return 1, "MPI_INT", True
    if isinstance(loma_ty, Array):
        cnt, prim, ok = get_flatten_info(loma_ty.t)
        return (loma_ty.static_size * cnt, prim, ok)
    if isinstance(loma_ty, Struct):
        total = 0
        prims = []
        ok_all = True
        for mem in loma_ty.members:
            cnt, prim, ok = get_flatten_info(mem.t)
            total += cnt if cnt else 0
            prims.append(prim)
            ok_all &= ok
        if ok_all and all(p == prims[0] for p in prims):
            return total, prims[0], True
    return None, None, False

def emit_mpi_type_definition(s: loma_ir.Struct) -> str:
    """
    Generate C code that creates and commits an MPI_Datatype for struct 's'.

    Returns a string like:
      MPI_Datatype mpi_t_StructName;
      MPI_Type_contiguous(...);
      MPI_Type_commit(&mpi_t_StructName);
    or, for heterogeneous structs:
      MPI_Datatype mpi_t_StructName;
      {
          int blocklens[N] = { ... };
          MPI_Aint displs[N] = { offsetof(...), ... };
          MPI_Datatype types[N] = { ..., ... };
          MPI_Type_create_struct(N, blocklens, displs, types, &mpi_t_StructName);
      }
      MPI_Type_commit(&mpi_t_StructName);
    """
    name   = s.id
    mpivar = f"mpi_t_{name}"
    lines  = []

    # check if all members are floats or float-arrays -> contiguous
    all_float = True
    total_count = 0
    for m in s.members:
        t = m.t
        if isinstance(t, loma_ir.Float):
            total_count += 1
        elif isinstance(t, loma_ir.Array) and isinstance(t.t, loma_ir.Float):
            total_count += t.static_size
        else:
            all_float = False
    if all_float:
        # contiguous block of total_count MPI_FLOATs
        lines.append(f"MPI_Type_contiguous({total_count}, MPI_FLOAT, &{mpivar});")
    else:
        n = len(s.members)
        # block lengths
        bls = ", ".join(
            str(m.t.static_size if isinstance(m.t, loma_ir.Array) else 1)
            for m in s.members
        )
        # displacements
        disps = ", ".join(f"offsetof({name},{m.id})" for m in s.members)
        types = []
        for m in s.members:
            t = m.t
            if isinstance(t, loma_ir.Float):
                types.append("MPI_FLOAT")
            elif isinstance(t, loma_ir.Int):
                types.append("MPI_INT")
            elif isinstance(t, loma_ir.Array) and isinstance(t.t, loma_ir.Float):
                types.append("MPI_FLOAT")
            elif isinstance(t, loma_ir.Array) and isinstance(t.t, loma_ir.Int):
                types.append("MPI_INT")
            elif isinstance(t, loma_ir.Struct):
                types.append(f"mpi_t_{t.id}")
            else:
                types.append("MPI_BYTE")
        types_list = ", ".join(types)

        lines.append("{")
        lines.append(f"    int blocklens[{n}] = {{ {bls} }};")
        lines.append(f"    MPI_Aint displs[{n}] = {{ {disps} }};")
        lines.append(f"    MPI_Datatype types[{n}] = {{ {types_list} }};")
        lines.append(f"    MPI_Type_create_struct({n}, blocklens, displs, types, &{mpivar});")
        lines.append("}")

    lines.append(f"MPI_Type_commit(&{mpivar});")
    return "\n".join(lines)
