M1 22 3 0 0 NMOS
   * M1: Drain at node 22, Gate at 3, Source at 0 (ground), 0 (body connected to source)
   
   * Note: Typically, in SPICE, transistor parameters would follow the instantiation.
   * Add specific NMOS model and parameter values as required for simulation.
   
   RO 22 0 ro_value
   * RO: Drain resistor from node 22 to ground, ro_value needs specification