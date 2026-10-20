Logs



(rl) anubhav@r2-d2:~/development/iitd_assignments/rl/A3/Q3_SupplyRun$ cd Q3_SupplyRun/
bash: cd: Q3_SupplyRun/: No such file or directory
(rl) anubhav@r2-d2:~/development/iitd_assignments/rl/A3/Q3_SupplyRun$ python main.py 
MDP(S=1920, A=4, start=1708, gamma=0.99, T=200, min b>0 = 0.15)

== Behaviour-policy dataset ==
{
 "n_trajectories": 2000,
 "discounted_return": {
  "mean": -13.9977053409059,
  "std": 10.0860550479983,
  "min": -17.374916255379564,
  "max": 80.97278682212584
 },
 "undiscounted_return": {
  "mean": -17.94,
  "std": 15.526652519755656,
  "min": -20.0,
  "max": 100.0
 },
 "mean_length": 30.6525,
 "frac_terminated": 0.999,
 "frac_truncated": 0.001,
 "frac_positive_return": 0.017
}
exact V^b(s0) = -13.8194   vs  mean(G_i) = -13.9977 (+-0.226 s.e.)

== Offline learning (800 episodes, 10 seeds, alpha=0.1) ==
  QL                   final greedy return =   79.770 +- 1.242   (non-terminating seeds: 0)
  ExpSARSA(eps=0.05)   final greedy return =   70.409 +- 30.844   (non-terminating seeds: 0)
  ExpSARSA(eps=0.1)    final greedy return =   70.409 +- 30.844   (non-terminating seeds: 0)
  ExpSARSA(eps=0.2)    final greedy return =   70.409 +- 30.844   (non-terminating seeds: 0)
  ExpSARSA(eps=0.3)    final greedy return =   50.357 +- 46.770   (non-terminating seeds: 0)
  IS-SARSA(eps=0.05)   final greedy return =   11.494 +- 46.495   (non-terminating seeds: 0)
  IS-SARSA(eps=0.1)    final greedy return =   11.494 +- 46.495   (non-terminating seeds: 0)
  IS-SARSA(eps=0.2)    final greedy return =   11.494 +- 46.495   (non-terminating seeds: 0)
  IS-SARSA(eps=0.3)    final greedy return =   11.494 +- 46.495   (non-terminating seeds: 0)

== Target policies: IS estimates and ratio statistics ==
  QL|target_eps=0.05               truth= 55.678 OIS=    0.133 WIS= -7.018 PDIS=    0.133 | mean(rho)=0.00223 Var(rho)=0.0681 ESS/N=0.00202
  QL|target_eps=0.1                truth= 36.821 OIS=    1.540 WIS= -6.439 PDIS=    1.540 | mean(rho)=0.0347 Var(rho)=10.9 ESS/N=0.00234
  QL|target_eps=0.2                truth= 10.903 OIS=    4.946 WIS= -6.283 PDIS=    4.946 | mean(rho)=0.256 Var(rho)=276 ESS/N=0.00228
  QL|target_eps=0.3                truth= -3.580 OIS=    0.340 WIS= -7.708 PDIS=    0.340 | mean(rho)=0.488 Var(rho)=314 ESS/N=0.00453
  ExpSARSA(eps=0.05)               truth= 47.470 OIS=    0.137 WIS= -7.470 PDIS=    0.137 | mean(rho)=0.00197 Var(rho)=0.0677 ESS/N=0.00128
  ExpSARSA(eps=0.1)                truth= 27.890 OIS=    1.666 WIS= -6.012 PDIS=    1.666 | mean(rho)=0.0258 Var(rho)=10.3 ESS/N=0.00143
  ExpSARSA(eps=0.2)                truth=  0.315 OIS=   -0.965 WIS=-15.010 PDIS=   -0.965 | mean(rho)=0.0624 Var(rho)=9.06 ESS/N=0.00153
  ExpSARSA(eps=0.3)                truth=-11.999 OIS=   -2.145 WIS=-16.462 PDIS=   -2.145 | mean(rho)=0.129 Var(rho)=34.1 ESS/N=0.00155
  IS-SARSA(eps=0.05)               truth=  2.421 OIS=   -0.006 WIS=-16.918 PDIS=   -0.006 | mean(rho)=0.0004 Var(rho)=0.00286 ESS/N=0.000997
  IS-SARSA(eps=0.1)                truth= -4.305 OIS=   -0.011 WIS=-16.768 PDIS=   -0.011 | mean(rho)=0.000678 Var(rho)=0.00232 ESS/N=0.00119
  IS-SARSA(eps=0.2)                truth=-12.601 OIS=   -0.087 WIS=-17.279 PDIS=   -0.087 | mean(rho)=0.00504 Var(rho)=0.045 ESS/N=0.00187
  IS-SARSA(eps=0.3)                truth=-15.583 OIS=   -0.520 WIS=-17.336 PDIS=   -0.520 | mean(rho)=0.03 Var(rho)=0.831 ESS/N=0.00233

Done in 13.4s -> results/results.json, plots/
(rl) anubhav@r2-d2:~/development/iitd_assignments/rl/A3/Q3_SupplyRun$ 

# rl-assignment-3
