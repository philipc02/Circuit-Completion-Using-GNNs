spice
* SPICE Netlist for the Amplifier Circuit

V1 Vin 0 DC 0

C1 Vin 2 1uF
C2 2 2 1uF ; connecting to the same net for information continuity

R1 3 2 10k
R2 2 0 10k

* The op-amp model is generic and might need a specific model declaration
* XOPAMP Non-Inverting Op-Amp Model
* Pin order: Non-inverting input, Inverting input, Output
XOPAMP 2 3 2 OPAMP_MODEL

* OPAMP_MODEL can be replaced with a specific op-amp model as needed
* .model OPAMP_MODEL OPAMP()

.end