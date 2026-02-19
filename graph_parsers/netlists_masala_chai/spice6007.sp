spice
* MOSFETs
M1 4 5 1 1 N_MOS   * Q1: D=4, G=5, S=1, B=1 (node number indicates the connection to node 4, 5, 1 for drain, gate, source)
M2 4 3 6 6 P_MOS   * Q3: D=4, G=3, S=6, B=6 (node number indicates the connection to node 4, 3, 6 for drain, gate, source)

* Voltage Sources
V1 6 1 DC Vdd     * Power Supply: Vdd connected from node 6 to ground (node 1)