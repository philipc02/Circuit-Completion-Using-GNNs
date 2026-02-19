* Main Circuit Components
V1 2 3 DC VBC        * Voltage source VBC, connected between nodes 2 and 3
I1 2 4 DC alpha_IE_R * Current source alpha*IE*R, connected between nodes 2 and 4
D1 4 3 Dmodel        * Diode connected between nodes 4 and 3

.model Dmodel D(IS=1e-14) * Diode model