* Op-amp circuit
R1 2 3 10k
XOPAMP 3 2 4 OPAMP_MODEL

* Voltage source for virtual ground
V1 4 0 0

* Define the op-amp model (ideal)
.model OPAMP_MODEL OPAMP(GBW=1e6)

.END