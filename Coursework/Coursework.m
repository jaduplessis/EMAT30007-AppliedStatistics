% ---------------- Question 1(a) ----------------
% Original data
% Null hypothesis: The drug does not have a significant effect on the physiological variable
v_drugs20 = load('drugs20.txt'); 
v_placebo20 = load('placebo20.txt'); 
[h, p_value_20, ci, stats] = ttest2(v_drugs20, v_placebo20);
fprintf('The p value for the null hypothesis is: %s \n', p_value_20)
% p_value_20 = 5.016187e-13 << 0.05, so we reject the null hypothesis.
% The drug has a statistically significant effect on the physiological variable.

% Alternative hypothesis 1: The mean of drug group is 1.2 * the mean of the placebo group
expected_drug = 1.2 * mean(v_placebo20);
[h, p_value_increase, ci, stats] = ttest(v_drugs20, expected_drug);
fprintf('The p value for the alternative hypothesis 1 is: %s \n', p_value_increase)
% p_value_increase = 8.452449e-06 << 0.05, so we reject the alternative hypothesis 1
% The mean of the drug group is significantly different from 1.2 * the mean of the placebo group.

% Alternative hypothesis 2: The mean of drug group is 1.3 * the mean of the placebo group
expected_drug = 1.3 * mean(v_placebo20);
[h, p_value_increase, ci, stats] = ttest(v_drugs20, expected_drug);
fprintf('The p value for the alternative hypothesis 2 is: %s \n', p_value_increase)
% p_value_increase = 0.3244 >> 0.05, so we fail to reject the alternative hypothesis 2
% There is no significant difference between the mean of the drug group and 1.3 times the mean of
% the placebo group

% New data
% Null hypothesis: The drug does not have a significant effect on the physiological variable
v_drugs30 = load('drugs30.txt');
v_placebo30 = load('placebo30.txt'); 
[h, p_value_30, ci, stats] = ttest2(v_drugs30, v_placebo30, 'tail', 'right');
fprintf('The p value for the null hypothesis is %s \n', p_value_30)
% p_value_30 = 0.5913 >> 0.05, so we fail to reject the null hypothesis. 
% The drug has no statistically significant effect in increasing the physiological variable.

% ---------------- Question 1(b) ----------------
all_drugs = [v_drugs20; v_drugs30]; 
all_placebo = [v_placebo20; v_placebo30];
n_iterations = 10000;
alpha = 0.05;

bootstrap_estimates = zeros(n_iterations, 1);
% Sample and replace to create bootstrap estimations
for i = 1:n_iterations
    resampled_drug = datasample(all_drugs, 20);
    mean_drug = mean(resampled_drug);
    resampled_placebo = datasample(all_placebo, 20);
    mean_placebo = mean(resampled_placebo);
    bootstrap_estimates(i) = (mean_drug - mean_placebo) / mean_placebo * 100;
end
bCI = quantile(bootstrap_estimates, [alpha/2, 1-alpha/2]);
fprintf('95%% Confidence Interval: [%.2f, %.2f]\n',bCI(1), bCI(2));
% Confidence interval is [1.35, 21.03]. 95% sure true percentage difference lies in this interval.
% Both lower bound and upper bound above 0 suggesting that true effect of drug is positive.
% However, interval is quite large so the effect is uncertain: can be relatively small or quite 
% substantial. Bootstrapping relies on sampling original dataset which may not perfectly represent 
% underlying population. Only 40 samples used which may not represent the population well.
% Distribution is also centered on observed statistic and not the population paramater. It may not 
% be a good indicator of the change of mean value.

% ---------------- Question 1(c) ----------------
all_data = [all_drugs; all_placebo];

% Create custom distribution
mix_normal_pdf = @(x, mu1, sigma1, mu2, sigma2, p) ...
    p * normpdf(x, mu1, sigma1) + (1 - p) * normpdf(x, mu2, sigma2);

% Initial guesses for the parameters
mu1_init = mean(all_data) - 0.5 * std(all_data);
sigma1_init = std(all_data);
mu2_init = mean(all_data) + 0.5 * std(all_data);
sigma2_init = std(all_data);
p_init = 0.5;
init_params = [mu1_init, sigma1_init, mu2_init, sigma2_init, p_init];

% Fit the distribution using maximum likelihood estimation
params = mle(all_data, 'pdf', mix_normal_pdf, 'start', init_params);

% Create a gmdistribution object
mu = [params(1), params(3)]'; % means of the two normal distributions
sigma = cat(3, params(2)^2, params(4)^2); % covariance matrices
p = [params(5), 1 - params(5)]; % mixing proportions
gm = gmdistribution(mu, sigma, p);

% Plot all data
subplot(2,2,1)
x_values = linspace(4, 13, 1000)';
y_values = pdf(gm, x_values);
histogram(all_data, 'Normalization', 'pdf', 'NumBins',15)
hold on;
plot(x_values, y_values, 'LineWidth', 2);
title('Probability Density Function')
xlabel('Physiological Variable');
ylabel('PDF');
xlim([4, 13])
legend('Data', 'Fitted Distribution', 'Location', 'northoutside')

subplot(2,2,2)
exp_cdf = cdf(gm, x_values);
stairs(x_values, exp_cdf, '-r', 'LineWidth', 2)
hold on
[exp_ecdf, x] = ecdf(all_data);
stairs(x, exp_ecdf, '.-k', 'LineWidth',2)
legend('theoretical CDF', 'empirical CDF', 'Location','northoutside')
title('Cumulative Density Function') % title for plot
xlabel('Outcomes') % x-axis label
ylabel('CDF')

subplot(2,2,[3,4])
pp = 0: 0.01: 1;
samples = random(gm, 1000);
theor_q = quantile(samples, pp);
empir_q = quantile(all_data, pp);
plot(theor_q, empir_q, 's', 'MarkerSize', 10)
hold on
plot(theor_q, theor_q, '-','HandleVisibility','off')
title('Q-Q plot')
xlabel('Theoretical quantiles')
ylabel('Empirical quantiles') 
% ---------------- Question 1(d) ----------------

