* Netlist for the provided circuit

VDD 3 0 DC 5V       * Power supply

Vin 1 0             * Input voltage

RD 3 2 1k           * Resistor RD

M1 4 2 3 NMOS_MODEL * NMOS M1 (Drain=4, Gate=2, Source=3)

M2 2 1 4 NMOS_MODEL * NMOS M2 (Drain=2, Gate=1, Source=4)

I1 4 0 DC 1mA       * Current source I1

.model NMOS_MODEL NMOS (Level=1) * NMOS model parameter

* End of netlist