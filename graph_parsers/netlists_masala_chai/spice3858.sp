* BJT Amplifier Circuit

* Components
VBB 6 0 DC <value>
V1 2 0 AC <value>
RB 2 3 <RB_value>
RC 4 5 <RC_value>
Q1 5 3 0 NPN

* Netlist
* VBB connected between node 6 and ground (node 0)
* ΔvI connected between node 2 and ground (node 0)
* RB connected between nodes 2 and 3
* RC connected between nodes 4 and 5 (V+ at node 4)
* Q1: Collector at node 5, Base at node 3, Emitter at node 0

.end