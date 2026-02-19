spice
*C1 connected between vin and ground
C1 6 2 

*Voltage source Phi1 connected at node 3 and ground
VPHI1 3 2 DC 0

*Voltage source Phi2 connected at node 3 and ground
VPHI2 3 2 DC 0

*C2 connected between node 3 and node 4
C2 3 4 

*Feedback capacitor CF connected between nodes 3 and 5
CF 3 5 

*Operational Amplifier
*Inverting input connected to node 3
*Non-inverting input connected to ground (node 2)
*Output connected to node 5
XOPAMP 3 2 5 OPAMP_MODEL

*.MODEL OPAMP_MODEL <op-amp parameters>