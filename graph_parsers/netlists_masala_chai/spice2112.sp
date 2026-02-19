plaintext
* SPICE Netlist for the given circuit

V1 4 5 DC <Value>

M1 2 3 0 0 NMOS_MODEL

G1 2 4 2 0 <gm_value>
RrO1 2 0 <rO1_value>
RrO2 2 4 <rO2_value>

* NMOS Model
.model NMOS_MODEL NMOS (level=1)

*.end