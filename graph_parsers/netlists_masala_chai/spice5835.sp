spice
* SPICE netlist for schematic

Vsig 5 6 DC 0
Rsig 5 3 Rsig
Re 3 2 (beta+1)*re
RL 2 6 (beta+1)*RL

* Define voltage output node
* The output voltage Vo is across node 2 (vo+) and ground (vo-)
* Vo = v(2) with respect to ground.

.end