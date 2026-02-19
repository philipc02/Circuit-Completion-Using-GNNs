spice
I1 5 2 DC 1A ; Current Source from node 5 to node 2

Q1 5 4 3 QMODEL ; NPN Transistor with collector at node 5, base at node 4, emitter at node 3

.model QMODEL NPN