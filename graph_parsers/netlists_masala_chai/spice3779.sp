plaintext
* SPICE netlist for given circuit

V1 5 8 DC Vi
Ri 5 5 0
RSi 5 5 RSi
R1R2 5 5 R1R2
Gm1 3 2 VALUE={g_m1*(V(5,0)-V(3,0))}
Rd1 3 6 Rd1
Gm2 6 4 VALUE={g_m2*(V(2,0)-V(4,0))}
Rs2 2 4 Rs2
Rl 2 0 Rl

.END