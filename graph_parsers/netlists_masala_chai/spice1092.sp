spice
* NMOS Transistor 
M1 <Drain_Node> <Gate_Node> 0 0 NMOS

* Voltage Source
V1 <Gate_Node> 0 DC 1

* Model
.model NMOS NMOS (Level=1)