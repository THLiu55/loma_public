# Front Interface for MPI in loma

*  mpi_rank()
return the current rank()

* Scatter for MPI
```
scatter( global : In[Array[Int / Float]], local : Out[Array[Int / Float]], scatter_size : Int )
```

* Gather for MPI
```
mpi_gather( global : Out[Array], local : In[Array], gather_size : Int )
```

