spice
* CMOS Circuit
* Voltage Source
VDD VDD 0 DC 5V

* PMOS Transistor
* ML (Drain, Gate, Source, Body)
ML f 2 VDD VDD PMOS_MODEL

* NMOS Transistors
* MDA (Drain, Gate, Source, Body)
MDA 2 A 3 3 NMOS_MODEL

* MDB (Drain, Gate, Source, Body)
MDB 3 B 0 0 NMOS_MODEL

* MDA_BAR (Drain, Gate, Source, Body)
MDA_BAR 3 A_BAR 4 4 NMOS_MODEL

* MDB_BAR (Drain, Gate, Source, Body)
MDB_BAR f B_BAR 0 0 NMOS_MODEL

.model PMOS_MODEL PMOS (LEVEL=1)
.model NMOS_MODEL NMOS (LEVEL=1)

.end