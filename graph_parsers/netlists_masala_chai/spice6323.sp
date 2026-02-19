plaintext
* SPICE netlist for the given schematic

M1 3 1 2 2 NMOS_MODEL
M2 5 3 3 3 PMOS_MODEL

* Model definitions (example, need to define or include actual models)
.model NMOS_MODEL NMOS (Level=1)
.model PMOS_MODEL PMOS (Level=1)

.END