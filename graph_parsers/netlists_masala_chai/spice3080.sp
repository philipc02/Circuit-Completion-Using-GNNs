* SPICE netlist for given schematic

M1 3 4 0 0 NMOS   * M1: (Drain=3, Gate=4, Source=0)
M2 3 2 1 1 PMOS   * M2: (Drain=3, Gate=2, Source=1)

I0 1 3 DC 0.1A    * Current Source I0: (Positive=1, Negative=3)

CL 3 0 10pF       * Capacitor CL: (Connected between 3 and 0)

VDD 1 0 DC 5V     * Voltage Source VDD: (Positive=1, Ground=0)
Vb 2 0 DC 1.8V    * Bias Voltage Vb: (Connected to 2, Ground=0)
Vin 4 0           * Input Voltage Vin: (Connected to 4, Ground=0)