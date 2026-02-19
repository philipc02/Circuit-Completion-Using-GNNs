spice
*Transistors
Q3 N2 N4 N25 N25 NMOS
Q4 N2 N2 N25 N25 PMOS
Q9 N3 N6 N2 N2 NMOS
Q10 N3 N10 N2 N2 NMOS

*Current Sources
I1 N6 N4 6uA
I2 N6 N8 6uA
I3 N25 N12 6uA
I4 N2 N11 21uA

*Capacitors
C21 N2 N11 5.2pF
C22 N3 N10 5.5pF

*Resistors
R9 N11 N2 22k
R10 N11 N2 22k

*Voltage Source (Bias)
VBIASCM N8 N3 DC 0V

*Additional Labels
VCC N6 0 DC 'VCC'
VEE N25 0 DC '-VEE'
VIN N4 0 DC 'Vin'
VOUT N3 0 DC 'Vout'

*End of netlist