plaintext
* SPICE Netlist
V1 3 1 DC 'vi/2'
I1 3 2 DC 'ii/2'
RF 3 4  RF_value
* Operational Amplifier model (abstract model)
XOPAMP 3 4 2 2 OpAmp
.model OpAmp (abstract)