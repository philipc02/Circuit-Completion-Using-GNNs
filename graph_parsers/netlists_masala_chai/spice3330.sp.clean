// Voltage Sources
VDD 3 0 DC
VA 2 0 DC
VB 5 0 DC

// Transistors
M1 2 3 5 5 PMOS    // PMOS: Drain=2, Gate=3, Source=5, Body=5
M2 3 2 2 0 NMOS    // NMOS: Drain=3, Gate=2, Source=2, Body=0
M3 33 3 0 0 NMOS   // NMOS: Drain=33, Gate=3, Source=0, Body=0
M4 33 3 3 3 PMOS   // PMOS: Drain=33, Gate=3, Source=3, Body=3

// Resistor
R1 33 3  // Resistor connected between node 33 and VA-B (node 3)

// Capacitors
C1 4 0   // Capacitor connected between CK (4) and ground
C2 4 3   // Capacitor connected between Vin (4) and node 3

// Nodes
* Node 2: Connected to VA
* Node 3: Connected to Vout, also common node
* Node 5: Connected to VB
* Node 33: Connected to output node VA-B