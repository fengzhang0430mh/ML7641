
library(MDPtoolbox)


P <- array(0, c(2,2,2))
P[,,1] <- matrix(c(0.5, 0.5, 0.8, 0.2), 2, 2, byrow=TRUE)
P[,,2] <- matrix(c(0, 1, 0.1, 0.9), 2, 2, byrow=TRUE)
R<- matrix(c(5, 10, -1, 2), 2, 2, byrow=TRUE)
mdp_policy_iteration_modified(P, R, 0.9)


### four states
### two actions action1: not fish action2: fish

P <- array(0, c(4,4,2))
P[,,1] <- matrix(c(0, 1, 0, 0, 
                   0, 0.3, 0.7, 0,
                   0, 0, 0.25, 0.75,
                   0, 0, 0.05, 0.95), 4, 4, byrow=TRUE)

P[,,2] <- matrix(c(1, 0, 0, 0, 
                   0.75, 0.25, 0, 0,
                   0, 0.75, 0.25, 0,
                   0, 0, 0.6, 0.4), 4, 4, byrow=TRUE)


R <- matrix(c(0, -200, 0, 0,
             10, 5, 0, 0,
             0, 10, 10, 0,
             0, 0, 50, 50), 4, 4, byrow=TRUE)


mdp_policy_iteration(P, R, discount = 0.7)

## experiment 1

discount_exp <- seq(0.1,0.9,0.1)
value_exp1 <- rep(NA, length(discount_exp))
for (i in 1:9){
  ts <- mdp_policy_iteration(P, R, discount = discount_exp[i])
  value_exp1[i] <- sum(ts$V)
}
plot(discount_exp, value_exp1)


### Experiment 2

P <- array(0, c(4,4,2))
P[,,1] <- matrix(c(0, 1, 0, 0, 
                   0, 0.3, 0.7, 0,
                   0, 0, 0.25, 0.75,
                   0, 0, 0.05, 0.95), 4, 4, byrow=TRUE)

P[,,2] <- matrix(c(1, 0, 0, 0, 
                   0.75, 0.25, 0, 0,
                   0, 0.75, 0.25, 0,
                   0, 0, 0.6, 0.4), 4, 4, byrow=TRUE)


R <- matrix(c(0, 0, 0, 0,
              10, 25, 0, 0,
              0, 10, 30, 0,
              0, 0, 50, 50), 4, 4, byrow=TRUE)

mdp_policy_iteration(P, R, discount = 0.8)


#### value iteration

mdp_value_iteration(P, R, 0.9, epsilon=0.05, max_iter=1000)


discount_exp <- seq(0.1,0.9,0.1)
value_exp1 <- rep(NA, length(discount_exp))
for (i in 1:9){
  ts <- mdp_value_iteration(P, R, discount_exp[i], epsilon=0.05, max_iter=1000)
  value_exp1[i] <- sum(ts$V)
}
plot(discount_exp, value_exp1)


epi_exp <- seq(0.01, 0.5, 0.01)
for (i in 1:length(epi_exp)){
  ts <- mdp_value_iteration(P, R, 0.9, epsilon=epi_exp[i], max_iter=1000)
  value_exp1[i] <- ts$iter
}
plot(epi_exp, value_exp1)



discount_exp <- seq(0.05,0.9,0.05)
value_exp1 <- rep(NA, length(discount_exp))

for (i in 1:length(discount_exp)){
  ts <- mdp_value_iteration(P, R, discount = discount_exp[i], epsilon=0.05, max_iter=1000)
  value_exp1[i] <- ts$time
}
plot(discount_exp, value_exp1)



### Q - learning

mdp_Q_learning(P, R, discount = 0.9)

discount_exp <- seq(0.1, 0.9, 0.05)
value_exp1 <- rep(NA, length(discount_exp))
for (i in 1:length(discount_exp)){
  ts <- mdp_Q_learning(P, R, discount = discount_exp[i])
  value_exp1[i] <- sum(ts$V)
}
plot(discount_exp, value_exp1)


x2 = mdp_Q_learning(P, R, discount = 0.8)
plot(as.ts(x2$mean_discrepancy))


### Question 2
### Quiz Game Show
### 10 states and 2 actions: 1 is play, 2 is quit

dat <- mdp_example_rand(S = 500, A = 10, is_sparse = T)

discount_exp <- seq(0.1,0.9,0.1)
value_exp1 <- rep(NA, length(discount_exp))

for (i in 1:9){
  ts <- mdp_policy_iteration(dat$P, dat$R, discount = discount_exp[i])
  value_exp1[i] <- sum(ts$V)
}
plot(discount_exp, value_exp1)

### experiment 2

discount_exp <- seq(0.1,0.9,0.1)
time_spent <- rep(NA, length(discount_exp))

for (i in 1:9){
  ts <- mdp_policy_iteration(dat$P, dat$R, discount = discount_exp[i])
  time_spent[i] <- sum(ts$time)
}
plot(discount_exp, time_spent)


### experiment 3

discount_exp <- seq(0.1,0.9,0.1)
itera_spent <- rep(NA, length(discount_exp))

for (i in 1:9){
  ts <- mdp_policy_iteration(dat$P, dat$R, discount = discount_exp[i])
  itera_spent[i] <- sum(ts$iter)
}
plot(discount_exp, itera_spent)

### value iteration

P <- dat$P
R <- dat$R
mdp_value_iteration(P, R, 0.9, epsilon=0.05, max_iter=1000)


discount_exp <- seq(0.1,0.9,0.1)
value_exp1 <- rep(NA, length(discount_exp))
for (i in 1:9){
  ts <- mdp_value_iteration(P, R, discount_exp[i], epsilon=0.05, max_iter=1000)
  value_exp1[i] <- sum(ts$V)
}
plot(discount_exp, value_exp1)


epi_exp <- seq(0.01, 0.5, 0.01)
for (i in 1:length(epi_exp)){
  ts <- mdp_value_iteration(P, R, 0.9, epsilon=epi_exp[i], max_iter=1000)
  value_exp1[i] <- ts$time
}
plot(epi_exp, value_exp1)



discount_exp <- seq(0.05,0.9,0.05)
value_exp1 <- rep(NA, length(discount_exp))

for (i in 1:length(discount_exp)){
  ts <- mdp_value_iteration(P, R, discount = discount_exp[i], epsilon=0.05, max_iter=1000)
  value_exp1[i] <- ts$time
}
plot(discount_exp, value_exp1)


### Q learning

mdp_Q_learning(P, R, discount = 0.9)

discount_exp <- seq(0.1, 0.9, 0.1)
value_exp1 <- rep(NA, length(discount_exp))
for (i in 1:length(discount_exp)){
  ts <- mdp_Q_learning(P, R, discount = discount_exp[i])
  value_exp1[i] <- sum(ts$V)
}
plot(discount_exp, value_exp1)


x2 = mdp_Q_learning(P, R, discount = 0.5)
plot(as.ts(x2$mean_discrepancy))


