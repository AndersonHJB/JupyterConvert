"""
# Fall 2024 Final Exam Problem 1
"""
"""
## 1. 数据加载与预处理
"""
import pandas as pd

titanic_data_path = 'https://github.com/AndersonHJB/AndersonHJB.github.io/releases/download/V0.05/04-6420_titanic-1.csv'
data = pd.read_csv(titanic_data_path)
data.head()
# 检查缺失值情况
missing_values = data.isnull().sum()
missing_values
from sklearn.preprocessing import StandardScaler


# 编码分类变量
data_encoded = pd.get_dummies(data, columns=["Sex", "Pclass", "Embarked"], drop_first=True)

# 标准化数值变量
scaler = StandardScaler()
data_encoded["Age_scaled"] = scaler.fit_transform(data_encoded[["Age"]])
data_encoded["Fare_scaled"] = scaler.fit_transform(data_encoded[["Fare"]])

# 删除原始数值列
data_encoded = data_encoded.drop(columns=["Age", "Fare"])


data_encoded.head()
# 检查缺失值
missing_values = data_encoded.isnull().sum()
print("缺失值统计：\n", missing_values)
"""
## 2. 模型定义与拟合
"""
import pymc as pm
import numpy as np

# 提取目标变量和特征
survived = data_encoded["Survived"].values
features = data_encoded[
    [
        "Sex_male",
        "Pclass_2",
        "Pclass_3",
        "Embarked_Q",
        "Embarked_S",
        "Age_scaled",
        "Fare_scaled",
        "SibSp",
        "Parch",
    ]
].values

# 标记缺失值
missing_age_indices = np.isnan(data["Age"].values)
age_observed = data_encoded["Age_scaled"][~missing_age_indices].to_numpy()

# 确保数据是数值类型
features = features.astype("float64")
age_observed = age_observed.astype("float64")

# PyMC 模型
with pm.Model() as logistic_model:
    # 截距和回归系数
    alpha = pm.Normal("alpha", mu=0, sigma=10)  # 截距
    betas = pm.Normal("betas", mu=0, sigma=2.5, shape=features.shape[1])  # 回归系数

    # 处理缺失值
    age_missing = pm.Uniform("age_missing", lower=0, upper=100, shape=int(np.sum(missing_age_indices)))
    age_combined = pm.math.concatenate([age_observed, age_missing])

    # 构建符号特征矩阵
    symbolic_features = pm.math.concatenate(
        [features[:, :5], age_combined[:, None], features[:, 6:]], axis=1
    )

    # 线性预测器
    eta = alpha + pm.math.dot(symbolic_features, betas)
    p = pm.Deterministic("p", pm.math.sigmoid(eta))

    # 观测值
    y_obs = pm.Bernoulli("y_obs", p=p, observed=survived)

    # 采样
    trace = pm.sample(1000, tune=1000, return_inferencedata=True, random_seed=42)

# 查看采样结果
print(pm.summary(trace))
# import pymc as pm
# import scipy
# print("PyMC version:", pm.__version__)
# print("Scipy version:", scipy.__version__)

# import pymc as pm
# import pytensor
# print("PyMC version:", pm.__version__)
# print("PyTensor version:", pytensor.__version__)

"""
## 3. 计算个例生存概率
"""
# 个例数据
jack_features = np.array([1, 0, 1, 0, 1, (20 - data["Age"].mean()) / data["Age"].std(), 
                          (10 - data["Fare"].mean()) / data["Fare"].std(), 0, 0])
rose_features = np.array([0, 0, 0, 0, 1, (17 - data["Age"].mean()) / data["Age"].std(), 
                          (130 - data["Fare"].mean()) / data["Fare"].std(), 0, 1])

# 计算生存概率
with logistic_model:
    jack_eta = alpha + pm.math.dot(jack_features, betas)
    jack_p = pm.math.sigmoid(jack_eta).eval()

    rose_eta = alpha + pm.math.dot(rose_features, betas)
    rose_p = pm.math.sigmoid(rose_eta).eval()

print(f"Jack 生存概率: {jack_p}")
print(f"Rose 生存概率: {rose_p}")

"""
## 4. 替代链接函数 (loglog)
"""
import arviz as az
with pm.Model() as loglog_model:
    # 定义模型的部分保持不变
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    betas = pm.Normal("betas", mu=0, sigma=2.5, shape=features.shape[1])

    # 处理缺失值
    age_missing = pm.Uniform("age_missing", lower=0, upper=100, shape=int(np.sum(missing_age_indices)))
    age_combined = pm.math.concatenate([age_observed, age_missing])

    symbolic_features = pm.math.concatenate(
        [features[:, :5], age_combined[:, None], features[:, 6:]], axis=1
    )

    eta = alpha + pm.math.dot(symbolic_features, betas)
    p = pm.Deterministic("p", pm.math.exp(-pm.math.exp(-eta)))

    # 定义 log_likelihood 显式变量
    log_likelihood = pm.Bernoulli("log_likelihood", p=p, observed=survived)

    # 使用 return_inferencedata=True 并指定 log_likelihood
    trace_loglog = pm.sample(
        1000, tune=1000, return_inferencedata=True, random_seed=42,
        idata_kwargs={"log_likelihood": True}
    )

# 计算 WAIC
waic_loglog = az.waic(trace_loglog)
print(f"Loglog 模型 WAIC: {waic_loglog}")

# import pandas as pd
# import numpy as np
# import pymc as pm
# import arviz as az
# import pytensor.tensor as at

# #---------------------------
# # 数据加载与预处理
# #---------------------------
# data_url = 'https://github.com/AndersonHJB/AndersonHJB.github.io/releases/download/V0.05/04-6420_titanic-1.csv'
# data = pd.read_csv(data_url)

# # 性别编码：female=1, male=0 (male为参考类)
# data['Sex'] = (data['Sex'] == 'female').astype(int)

# # Pclass: 1为参考类，生成Pclass_2, Pclass_3
# data = pd.get_dummies(data, columns=["Pclass"], prefix="Pclass", drop_first=True)

# # Embarked: C为参考类，生成Embarked_Q, Embarked_S
# data = pd.get_dummies(data, columns=["Embarked"], prefix="Embarked", drop_first=True)

# age_data = data["Age"].values
# fare_data = data["Fare"].values
# survived = data["Survived"].values

# Sex_female = data["Sex"].values      # female=1, male=0
# Pclass_2 = data["Pclass_2"].values
# Pclass_3 = data["Pclass_3"].values
# Embarked_Q = data["Embarked_Q"].values
# Embarked_S = data["Embarked_S"].values
# SibSp = data["SibSp"].astype(float).values
# Parch = data["Parch"].astype(float).values

# # 计算Age和Fare的均值与标准差（Age仅对已知值）
# observed_ages = age_data[~np.isnan(age_data)]
# age_mean = observed_ages.mean()
# age_std = observed_ages.std()

# fare_mean = fare_data.mean()
# fare_std = fare_data.std()
# fare_scaled = (fare_data - fare_mean)/fare_std

# missing_age_indices = np.isnan(age_data)
# n_missing = np.sum(missing_age_indices)

# #---------------------------
# # Logistic 回归模型
# #---------------------------
# with pm.Model() as logistic_model:
#     # 缺失Age用Uniform(0,100)先验，注意shape需为int类型
#     age_missing_raw = pm.Uniform("age_missing_raw", lower=0, upper=100, shape=int(n_missing))
#     # 将缺失值插入到age_data中
#     age_full_imp = at.as_tensor_variable(age_data)
#     age_full_imp = at.set_subtensor(age_full_imp[missing_age_indices], age_missing_raw)
#     # 标准化Age
#     age_full_scaled = (age_full_imp - age_mean)/age_std

#     # 先验分布
#     alpha = pm.Normal("alpha", mu=0, sigma=10)
#     # 特征顺序:
#     # [Sex_female, Pclass_2, Pclass_3, Embarked_Q, Embarked_S, Age_scaled, SibSp, Parch, Fare_scaled]
#     betas = pm.Normal("betas", mu=0, sigma=2.5, shape=9)

#     # 线性预测器
#     eta = (alpha
#            + betas[0]*Sex_female
#            + betas[1]*Pclass_2
#            + betas[2]*Pclass_3
#            + betas[3]*Embarked_Q
#            + betas[4]*Embarked_S
#            + betas[5]*age_full_scaled
#            + betas[6]*SibSp
#            + betas[7]*Parch
#            + betas[8]*fare_scaled)

#     p = pm.Deterministic("p", pm.math.sigmoid(eta))

#     # 观测数据
#     y_obs = pm.Bernoulli("y_obs", p=p, observed=survived)

#     # MCMC采样，增加log_likelihood存储
#     trace_logit = pm.sample(draws=2000, tune=2000, target_accept=0.9, random_seed=42, 
#                             return_inferencedata=True, 
#                             idata_kwargs={"log_likelihood": True})

# print("Logistic模型后验摘要：")
# print(az.summary(trace_logit, var_names=["alpha","betas"]))

# #---------------------------
# # 问题2: Jack和Rose生存概率计算
# #---------------------------
# # Jack: Age=20,male=0,三等舱(Pclass_3=1),S=1,Q=0,Fare=10,SibSp=0,Parch=0
# jack_age_scaled = (20 - age_mean)/age_std
# jack_fare_scaled = (10 - fare_mean)/fare_std
# jack_features = np.array([0,      # Sex_female=0
#                           0,      # Pclass_2=0
#                           1,      # Pclass_3=1
#                           0,      # Embarked_Q=0
#                           1,      # Embarked_S=1
#                           jack_age_scaled,
#                           0,      # SibSp=0
#                           0,      # Parch=0
#                           jack_fare_scaled])

# # Rose: Age=17,female=1,一等舱参考(Pclass_2=0,Pclass_3=0),S=1,Q=0,Fare=130,Parch=1,SibSp=0
# rose_age_scaled = (17 - age_mean)/age_std
# rose_fare_scaled = (130 - fare_mean)/fare_std
# rose_features = np.array([1,      # Sex_female=1
#                           0,      # Pclass_2=0
#                           0,      # Pclass_3=0
#                           0,      # Embarked_Q=0
#                           1,      # Embarked_S=1
#                           rose_age_scaled,
#                           0,      # SibSp=0
#                           1,      # Parch=1
#                           rose_fare_scaled])

# posterior = trace_logit.posterior
# alpha_samples = posterior["alpha"].values
# betas_samples = posterior["betas"].values

# def compute_prob(alpha_samp, betas_samp, features):
#     eta_val = alpha_samp + np.dot(betas_samp, features)
#     return 1/(1+np.exp(-eta_val))

# jack_probs = []
# rose_probs = []
# for i in range(alpha_samples.shape[0]):
#     for j in range(alpha_samples.shape[1]):
#         jack_probs.append(compute_prob(alpha_samples[i,j], betas_samples[i,j,:], jack_features))
#         rose_probs.append(compute_prob(alpha_samples[i,j], betas_samples[i,j,:], rose_features))

# jack_mean_prob = np.mean(jack_probs)
# rose_mean_prob = np.mean(rose_probs)
# print(f"Jack的生存概率均值: {jack_mean_prob:.4f}")
# print(f"Rose的生存概率均值: {rose_mean_prob:.4f}")

# #---------------------------
# # 问题3：使用loglog链接函数
# # p = exp(-exp(-eta))
# #---------------------------
# with pm.Model() as loglog_model:
#     age_missing_raw_ll = pm.Uniform("age_missing_raw_ll", lower=0, upper=100, shape=int(n_missing))
#     age_full_imp_ll = at.as_tensor_variable(age_data)
#     age_full_imp_ll = at.set_subtensor(age_full_imp_ll[missing_age_indices], age_missing_raw_ll)
#     age_full_scaled_ll = (age_full_imp_ll - age_mean)/age_std

#     alpha_ll = pm.Normal("alpha_ll", mu=0, sigma=10)
#     betas_ll = pm.Normal("betas_ll", mu=0, sigma=2.5, shape=9)

#     eta_ll = (alpha_ll
#               + betas_ll[0]*Sex_female
#               + betas_ll[1]*Pclass_2
#               + betas_ll[2]*Pclass_3
#               + betas_ll[3]*Embarked_Q
#               + betas_ll[4]*Embarked_S
#               + betas_ll[5]*age_full_scaled_ll
#               + betas_ll[6]*SibSp
#               + betas_ll[7]*Parch
#               + betas_ll[8]*fare_scaled)

#     p_loglog = pm.Deterministic("p_loglog", pm.math.exp(-pm.math.exp(-eta_ll)))
#     y_obs_ll = pm.Bernoulli("y_obs_ll", p=p_loglog, observed=survived)

#     trace_loglog = pm.sample(draws=2000, tune=2000, target_accept=0.9, random_seed=42, 
#                              return_inferencedata=True,
#                              idata_kwargs={"log_likelihood": True})

# print("Loglog模型后验摘要：")
# print(az.summary(trace_loglog, var_names=["alpha_ll","betas_ll"]))

# # 使用WAIC比较模型拟合度
# waic_logit = az.waic(trace_logit)
# waic_loglog = az.waic(trace_loglog)

# print(f"Logit模型 WAIC: {waic_logit.waic:.2f}")
# print(f"Loglog模型 WAIC: {waic_loglog.waic:.2f}")
# 使用中文注释，完整代码实现要求
# 确保： 
# 1. 使用贝叶斯方法（PPL）且无频率学派方法
# 2. 对Sex, Pclass, Embarked做哑变量编码，使用参考类别（male、Pclass=1、Embarked=C）
# 3. Age有缺失，用Uniform(0,100)先验进行缺失值插补，并对Age和Fare标准化（用1个标准差）
# 4. 拟合logistic回归模型，查看95%可信区间判断显著性
# 5. 给定场景(Jack和Rose)计算生存概率平均值
# 6. 用loglog链接函数重新拟合，并通过WAIC对比模型优劣

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as at

#---------------------------
# 数据加载与预处理
#---------------------------
data_url = 'https://github.com/AndersonHJB/AndersonHJB.github.io/releases/download/V0.05/04-6420_titanic-1.csv'
data = pd.read_csv(data_url)

# 性别编码：female=1, male=0 (male为参考类)
data['Sex'] = (data['Sex'] == 'female').astype(int)

# Pclass: 1为参考类，生成Pclass_2, Pclass_3
data = pd.get_dummies(data, columns=["Pclass"], prefix="Pclass", drop_first=True)

# Embarked: C为参考类，生成Embarked_Q, Embarked_S
data = pd.get_dummies(data, columns=["Embarked"], prefix="Embarked", drop_first=True)

age_data = data["Age"].values
fare_data = data["Fare"].values
survived = data["Survived"].values

Sex_female = data["Sex"].values      # female=1, male=0
Pclass_2 = data["Pclass_2"].values
Pclass_3 = data["Pclass_3"].values
Embarked_Q = data["Embarked_Q"].values
Embarked_S = data["Embarked_S"].values
SibSp = data["SibSp"].astype(float).values
Parch = data["Parch"].astype(float).values

# 计算Age和Fare的均值与标准差（Age仅对已知值）
observed_ages = age_data[~np.isnan(age_data)]
age_mean = observed_ages.mean()
age_std = observed_ages.std()

fare_mean = fare_data.mean()
fare_std = fare_data.std()
fare_scaled = (fare_data - fare_mean)/fare_std

missing_age_indices = np.isnan(age_data)
n_missing = np.sum(missing_age_indices)

#---------------------------
# Logistic 回归模型
#---------------------------
with pm.Model() as logistic_model:
    # 缺失Age用Uniform(0,100)先验，注意shape需为int类型
    age_missing_raw = pm.Uniform("age_missing_raw", lower=0, upper=100, shape=int(n_missing))
    # 将缺失值插入到age_data中
    age_full_imp = at.as_tensor_variable(age_data)
    age_full_imp = at.set_subtensor(age_full_imp[missing_age_indices], age_missing_raw)
    # 标准化Age
    age_full_scaled = (age_full_imp - age_mean)/age_std

    # 先验分布
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    # 特征顺序:
    # [Sex_female, Pclass_2, Pclass_3, Embarked_Q, Embarked_S, Age_scaled, SibSp, Parch, Fare_scaled]
    betas = pm.Normal("betas", mu=0, sigma=2.5, shape=9)

    # 线性预测器
    eta = (alpha
           + betas[0]*Sex_female
           + betas[1]*Pclass_2
           + betas[2]*Pclass_3
           + betas[3]*Embarked_Q
           + betas[4]*Embarked_S
           + betas[5]*age_full_scaled
           + betas[6]*SibSp
           + betas[7]*Parch
           + betas[8]*fare_scaled)

    p = pm.Deterministic("p", pm.math.sigmoid(eta))

    # 观测数据
    y_obs = pm.Bernoulli("y_obs", p=p, observed=survived)

    # MCMC采样，增加log_likelihood存储
    trace_logit = pm.sample(draws=2000, tune=2000, target_accept=0.9, random_seed=42, 
                            return_inferencedata=True, 
                            idata_kwargs={"log_likelihood": True})

print("Logistic模型后验摘要：")
print(az.summary(trace_logit, var_names=["alpha","betas"]))

#---------------------------
# 问题2: Jack和Rose生存概率计算
#---------------------------
# Jack: Age=20,male=0,三等舱(Pclass_3=1),S=1,Q=0,Fare=10,SibSp=0,Parch=0
jack_age_scaled = (20 - age_mean)/age_std
jack_fare_scaled = (10 - fare_mean)/fare_std
jack_features = np.array([0,      # Sex_female=0
                          0,      # Pclass_2=0
                          1,      # Pclass_3=1
                          0,      # Embarked_Q=0
                          1,      # Embarked_S=1
                          jack_age_scaled,
                          0,      # SibSp=0
                          0,      # Parch=0
                          jack_fare_scaled])

# Rose: Age=17,female=1,一等舱(Pclass=1参考),S=1,Q=0,Fare=130,Parch=1,SibSp=0
rose_age_scaled = (17 - age_mean)/age_std
rose_fare_scaled = (130 - fare_mean)/fare_std
rose_features = np.array([1,      # Sex_female=1
                          0,      # Pclass_2=0(一等舱参考)
                          0,      # Pclass_3=0
                          0,      # Embarked_Q=0
                          1,      # Embarked_S=1
                          rose_age_scaled,
                          0,      # SibSp=0
                          1,      # Parch=1
                          rose_fare_scaled])

posterior = trace_logit.posterior
alpha_samples = posterior["alpha"].values
betas_samples = posterior["betas"].values

def compute_prob(alpha_samp, betas_samp, features):
    eta_val = alpha_samp + np.dot(betas_samp, features)
    return 1/(1+np.exp(-eta_val))

jack_probs = []
rose_probs = []
for i in range(alpha_samples.shape[0]):
    for j in range(alpha_samples.shape[1]):
        jack_probs.append(compute_prob(alpha_samples[i,j], betas_samples[i,j,:], jack_features))
        rose_probs.append(compute_prob(alpha_samples[i,j], betas_samples[i,j,:], rose_features))

jack_mean_prob = np.mean(jack_probs)
rose_mean_prob = np.mean(rose_probs)
print(f"Jack的生存概率均值: {jack_mean_prob:.4f}")
print(f"Rose的生存概率均值: {rose_mean_prob:.4f}")

#---------------------------
# 问题3：使用loglog链接函数
# p = exp(-exp(-eta))
#---------------------------
with pm.Model() as loglog_model:
    age_missing_raw_ll = pm.Uniform("age_missing_raw_ll", lower=0, upper=100, shape=int(n_missing))
    age_full_imp_ll = at.as_tensor_variable(age_data)
    age_full_imp_ll = at.set_subtensor(age_full_imp_ll[missing_age_indices], age_missing_raw_ll)
    age_full_scaled_ll = (age_full_imp_ll - age_mean)/age_std

    alpha_ll = pm.Normal("alpha_ll", mu=0, sigma=10)
    betas_ll = pm.Normal("betas_ll", mu=0, sigma=2.5, shape=9)

    eta_ll = (alpha_ll
              + betas_ll[0]*Sex_female
              + betas_ll[1]*Pclass_2
              + betas_ll[2]*Pclass_3
              + betas_ll[3]*Embarked_Q
              + betas_ll[4]*Embarked_S
              + betas_ll[5]*age_full_scaled_ll
              + betas_ll[6]*SibSp
              + betas_ll[7]*Parch
              + betas_ll[8]*fare_scaled)

    p_loglog = pm.Deterministic("p_loglog", pm.math.exp(-pm.math.exp(-eta_ll)))
    y_obs_ll = pm.Bernoulli("y_obs_ll", p=p_loglog, observed=survived)

    trace_loglog = pm.sample(draws=2000, tune=2000, target_accept=0.9, random_seed=42, 
                             return_inferencedata=True,
                             idata_kwargs={"log_likelihood": True})

print("Loglog模型后验摘要：")
print(az.summary(trace_loglog, var_names=["alpha_ll","betas_ll"]))

# 使用WAIC比较模型，注意从返回对象中提取WAIC需要转换
waic_logit = az.waic(trace_logit)
waic_loglog = az.waic(trace_loglog)

# `az.waic()` 返回ELPDData对象，通过elpd_waic计算WAIC：WAIC = -2 * elpd_waic
print(f"Logit模型 WAIC: {-2 * waic_logit.elpd_waic:.2f}")
print(f"Loglog模型 WAIC: {-2 * waic_loglog.elpd_waic:.2f}")

"""
# Fall 2024 Final Exam Problem 2
"""
"""
## 第 1 部分：导入依赖和加载数据
"""
# https://github.com/AndersonHJB/AndersonHJB.github.io/releases/download/V0.05/06-torque-1.csv
import pandas as pd
import pymc as pm
import arviz as az
import numpy as np

# 加载数据
file_path = 'https://github.com/AndersonHJB/AndersonHJB.github.io/releases/download/V0.05/06-torque-1.csv'  # 请确保文件路径正确
data = pd.read_csv(file_path)

# 将分类变量编码
plating_levels = pd.Categorical(data['plating']).codes
medium_levels = pd.Categorical(data['medium']).codes
torque_values = data['torque'].values

"""
## 第 2 部分：两传统 ANOVA 分析
"""
import statsmodels.api as sm
from statsmodels.formula.api import ols

# 方差分析
model = ols('torque ~ C(plating) * C(medium)', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# 显示 ANOVA 结果
print("Two-Way ANOVA Results:")
print(anova_table)
"""
## 第 3 部分：贝叶斯模型（方差不随涂层类型变化）
"""
# 固定方差模型
with pm.Model() as fixed_variance_model:
    # 先验
    mu_intercept = pm.Normal("mu_intercept", mu=0, sigma=10)
    mu_plating = pm.Normal("mu_plating", mu=0, sigma=10, shape=len(set(plating_levels)))
    mu_medium = pm.Normal("mu_medium", mu=0, sigma=10, shape=len(set(medium_levels)))
    mu_interaction = pm.Normal("mu_interaction", mu=0, sigma=10, shape=(len(set(plating_levels)), len(set(medium_levels))))

    # 方差固定
    sigma = pm.HalfNormal("sigma", sigma=10)

    # 期望值
    mu = (
        mu_intercept +
        mu_plating[plating_levels] +
        mu_medium[medium_levels] +
        mu_interaction[plating_levels, medium_levels]
    )

    # 似然函数
    torque_obs = pm.Normal("torque_obs", mu=mu, sigma=sigma, observed=torque_values)

    # 采样
    # trace_fixed_variance = pm.sample(2000, tune=1000, return_inferencedata=True, cores=2)
    trace_fixed_variance = pm.sample(
        2000, 
        tune=1000, 
        return_inferencedata=True, 
        cores=2,
        idata_kwargs={"log_likelihood": True}  # 确保包含log_likelihood
    )

# 总结结果
print(az.summary(trace_fixed_variance, round_to=2))

"""
## 第 4 部分：贝叶斯模型（方差随涂层类型变化）
"""
# 可变方差模型
with pm.Model() as varying_variance_model:
    # 先验
    mu_intercept = pm.Normal("mu_intercept", mu=0, sigma=10)
    mu_plating = pm.Normal("mu_plating", mu=0, sigma=10, shape=len(set(plating_levels)))
    mu_medium = pm.Normal("mu_medium", mu=0, sigma=10, shape=len(set(medium_levels)))
    mu_interaction = pm.Normal("mu_interaction", mu=0, sigma=10, shape=(len(set(plating_levels)), len(set(medium_levels))))

    # 方差随涂层类型变化
    sigma_plating = pm.HalfNormal("sigma_plating", sigma=10, shape=len(set(plating_levels)))

    # 期望值
    mu = (
        mu_intercept +
        mu_plating[plating_levels] +
        mu_medium[medium_levels] +
        mu_interaction[plating_levels, medium_levels]
    )

    # 似然函数
    torque_obs = pm.Normal("torque_obs", mu=mu, sigma=sigma_plating[plating_levels], observed=torque_values)

    # 采样
    # trace_varying_variance = pm.sample(2000, tune=1000, return_inferencedata=True, cores=2)
    trace_varying_variance = pm.sample(
        2000, 
        tune=1000, 
        return_inferencedata=True, 
        cores=2,
        idata_kwargs={"log_likelihood": True}  # 确保包含log_likelihood
    )


# 总结结果
print(az.summary(trace_varying_variance, round_to=2))
"""
## 第 5 部分：比较模型性能（DIC/WAIC）
"""
import warnings

# 可以根据需要忽略特定的 UserWarning
warnings.filterwarnings("ignore", message="For one or more samples the posterior variance of the log predictive densities exceeds 0.4")

waic_fixed = az.waic(trace_fixed_variance)
waic_varying = az.waic(trace_varying_variance)

# 打印 elpd_waic
print("ELPD_WAIC for Fixed Variance Model:", waic_fixed.elpd_waic)
print("ELPD_WAIC for Varying Variance Model:", waic_varying.elpd_waic)

# 将 elpd_waic 转换成传统的 WAIC 指标（WAIC = -2 * elpd_waic）
fixed_waic_value = -2 * waic_fixed.elpd_waic
varying_waic_value = -2 * waic_varying.elpd_waic

print("WAIC for Fixed Variance Model:", fixed_waic_value)
print("WAIC for Varying Variance Model:", varying_waic_value)

# 模型选择
if fixed_waic_value < varying_waic_value:
    print("The fixed variance model performs better based on WAIC.")
else:
    print("The varying variance model performs better based on WAIC.")
import pandas as pd
import pymc as pm
import arviz as az
import numpy as np

# 加载数据
url = 'https://github.com/AndersonHJB/AndersonHJB.github.io/releases/download/V0.05/06-torque-1.csv'
data = pd.read_csv(url)

# 设置分类变量顺序
plating_cat = pd.Categorical(data['plating'], categories=["C&W","HT","P&O"])
medium_cat = pd.Categorical(data['medium'], categories=["M","B"])

plating_idx = plating_cat.codes
medium_idx = medium_cat.codes
torque_values = data['torque'].values

# 因素水平数
n_plating = 3
n_medium = 2

with pm.Model() as fixed_var_model:
    # 截距
    mu_intercept = pm.Normal("mu_intercept", mu=0, sigma=10)

    # Plating主效应参数化（STZ约束）
    alpha_CW = pm.Normal("alpha_CW", mu=0, sigma=10)
    alpha_HT = pm.Normal("alpha_HT", mu=0, sigma=10)
    alpha_PO = pm.Deterministic("alpha_PO", -(alpha_CW + alpha_HT))

    # Medium主效应参数化（STZ约束）
    beta_M = pm.Normal("beta_M", mu=0, sigma=10)
    beta_B = pm.Deterministic("beta_B", -beta_M)

    # 交互效应参数（STZ约束）
    gamma_CW_M = pm.Normal("gamma_CW_M", mu=0, sigma=10)
    gamma_CW_B = pm.Normal("gamma_CW_B", mu=0, sigma=10)
    gamma_HT_M = pm.Normal("gamma_HT_M", mu=0, sigma=10)
    gamma_HT_B = pm.Normal("gamma_HT_B", mu=0, sigma=10)

    gamma_PO_M = pm.Deterministic("gamma_PO_M", -(gamma_CW_M + gamma_HT_M))
    gamma_PO_B = pm.Deterministic("gamma_PO_B", -(gamma_CW_B + gamma_HT_B))

    # 使用pm.math.stack来创建张量而不是Python列表
    alpha_arr = pm.Deterministic("alpha_arr", pm.math.stack([alpha_CW, alpha_HT, alpha_PO]))
    beta_arr = pm.Deterministic("beta_arr", pm.math.stack([beta_M, beta_B]))

    row1 = pm.math.stack([gamma_CW_M, gamma_CW_B])
    row2 = pm.math.stack([gamma_HT_M, gamma_HT_B])
    row3 = pm.math.stack([gamma_PO_M, gamma_PO_B])
    gamma_matrix = pm.Deterministic("gamma_matrix", pm.math.stack([row1, row2, row3]))

    # 固定方差
    sigma = pm.HalfNormal("sigma", sigma=10)

    # 均值表达式
    mu = (mu_intercept 
          + alpha_arr[plating_idx] 
          + beta_arr[medium_idx]
          + gamma_matrix[plating_idx, medium_idx])

    # 似然
    torque_obs = pm.Normal("torque_obs", mu=mu, sigma=sigma, observed=torque_values)

    # 采样
    trace_fixed = pm.sample(2000, tune=1000, cores=2, return_inferencedata=True, 
                            idata_kwargs={"log_likelihood": True})


with pm.Model() as varying_var_model:
    mu_intercept = pm.Normal("mu_intercept", mu=0, sigma=10)

    # Plating主效应（STZ约束）
    alpha_CW = pm.Normal("alpha_CW", mu=0, sigma=10)
    alpha_HT = pm.Normal("alpha_HT", mu=0, sigma=10)
    alpha_PO = pm.Deterministic("alpha_PO", -(alpha_CW + alpha_HT))

    # Medium主效应（STZ约束）
    beta_M = pm.Normal("beta_M", mu=0, sigma=10)
    beta_B = pm.Deterministic("beta_B", -beta_M)

    # 交互效应（STZ约束）
    gamma_CW_M = pm.Normal("gamma_CW_M", mu=0, sigma=10)
    gamma_CW_B = pm.Normal("gamma_CW_B", mu=0, sigma=10)
    gamma_HT_M = pm.Normal("gamma_HT_M", mu=0, sigma=10)
    gamma_HT_B = pm.Normal("gamma_HT_B", mu=0, sigma=10)

    gamma_PO_M = pm.Deterministic("gamma_PO_M", -(gamma_CW_M + gamma_HT_M))
    gamma_PO_B = pm.Deterministic("gamma_PO_B", -(gamma_CW_B + gamma_HT_B))

    alpha_arr = pm.Deterministic("alpha_arr", pm.math.stack([alpha_CW, alpha_HT, alpha_PO]))
    beta_arr = pm.Deterministic("beta_arr", pm.math.stack([beta_M, beta_B]))

    row1 = pm.math.stack([gamma_CW_M, gamma_CW_B])
    row2 = pm.math.stack([gamma_HT_M, gamma_HT_B])
    row3 = pm.math.stack([gamma_PO_M, gamma_PO_B])
    gamma_matrix = pm.Deterministic("gamma_matrix", pm.math.stack([row1, row2, row3]))

    # 方差随Plating变化
    sigma_plating = pm.HalfNormal("sigma_plating", sigma=10, shape=n_plating)

    mu = (mu_intercept 
          + alpha_arr[plating_idx] 
          + beta_arr[medium_idx] 
          + gamma_matrix[plating_idx, medium_idx])

    torque_obs = pm.Normal("torque_obs", mu=mu, sigma=sigma_plating[plating_idx], observed=torque_values)
    
    trace_varying = pm.sample(2000, tune=1000, cores=2, return_inferencedata=True,
                              idata_kwargs={"log_likelihood": True})


print("固定方差模型参数总结：")
print(az.summary(trace_fixed, var_names=["mu_intercept","alpha_CW","alpha_HT","alpha_PO",
                                         "beta_M","beta_B","gamma_CW_M","gamma_CW_B",
                                         "gamma_HT_M","gamma_HT_B","gamma_PO_M","gamma_PO_B","sigma"], 
                 round_to=2))

print("\n方差随Plating变化模型参数总结：")
print(az.summary(trace_varying, var_names=["mu_intercept","alpha_CW","alpha_HT","alpha_PO",
                                           "beta_M","beta_B","gamma_CW_M","gamma_CW_B",
                                           "gamma_HT_M","gamma_HT_B","gamma_PO_M","gamma_PO_B","sigma_plating"], 
                 round_to=2))

# 使用WAIC比较模型
waic_fixed = az.waic(trace_fixed)
waic_varying = az.waic(trace_varying)

print("\nWAIC比较：")
print("Fixed Variance Model WAIC:", -2 * waic_fixed.elpd_waic)
print("Varying Variance Model WAIC:", -2 * waic_varying.elpd_waic)

if (-2 * waic_fixed.elpd_waic) < (-2 * waic_varying.elpd_waic):
    print("基于WAIC，固定方差模型表现更好。")
else:
    print("基于WAIC，方差随Plating变化的模型表现更好。")
"""
# Fall 2024 Final Exam Problem 3
"""
# import pymc as pm
# import numpy as np
# import pandas as pd
# import arviz as az
# from scipy.stats import norm

# # 读取数据
# data = pd.read_csv("https://github.com/AndersonHJB/AndersonHJB.github.io/releases/download/V0.05/05-nanowire-2.csv")
# x = data['x'].values
# y = data['y'].values

# # 构建概率模型
# with pm.Model() as model:
#     # 参数的先验分布
#     theta1 = pm.Lognormal('theta1', mu=0, sigma=10)
#     theta3 = pm.Lognormal('theta3', mu=0, sigma=10)
#     theta4 = pm.Lognormal('theta4', mu=0, sigma=10)
#     theta2 = pm.Uniform('theta2', lower=0, upper=1)

#     # 使用 erfc 实现标准正态分布的累积分布函数
#     norm_cdf = 0.5 * pm.math.erfc(-x / (theta4 * pm.math.sqrt(2)))

#     # 定义均值函数
#     mu = (
#         theta1 * pm.math.exp(-theta2 * x**2) +
#         theta3 * (1 - pm.math.exp(-theta2 * x**2)) * norm_cdf
#     )

#     # 观测数据的似然函数
#     y_obs = pm.Poisson('y_obs', mu=mu, observed=y)

#     # 采样设置：10,000个后验样本，1,000个调整样本
#     trace = pm.sample(10000, tune=1000, target_accept=0.95, random_seed=42)

# # 后验样本的统计摘要
# summary = az.summary(trace, hdi_prob=0.95)
# print(summary)

# # 使用后验分布预测厚度为 1.5 nm 时的纳米线密度
# x_new = 1.5
# theta1_samples = trace.posterior['theta1'].values.flatten()
# theta3_samples = trace.posterior['theta3'].values.flatten()
# theta4_samples = trace.posterior['theta4'].values.flatten()
# theta2_samples = trace.posterior['theta2'].values.flatten()

# # 构建新数据的均值
# norm_cdf_new = 0.5 * norm.cdf(-x_new / (theta4_samples * np.sqrt(2)))
# mu_new = (
#     theta1_samples * np.exp(-theta2_samples * x_new**2) +
#     theta3_samples * (1 - np.exp(-theta2_samples * x_new**2)) * norm_cdf_new
# )

# # 预测分布
# predictive_samples = np.random.poisson(mu_new)
# predictive_mean = np.mean(predictive_samples)
# predictive_hdi = az.hdi(predictive_samples, hdi_prob=0.95)

# # 输出预测结果
# print(f"Predictive mean: {predictive_mean}")
# print(f"95% HDI: {predictive_hdi}")
# -*- coding: utf-8 -*-
# @Time    : 2024/12/8 21:53
# @Author  : AI悦创
# @FileName: Q3.py
# @Software: PyCharm
# @Blog    ：https://bornforthis.cn/
# code is far away from bugs with the god animal protecting
#    I love animals. They taste delicious.
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
from scipy.stats import norm

# 读取数据
data = pd.read_csv("https://github.com/AndersonHJB/AndersonHJB.github.io/releases/download/V0.05/05-nanowire-2.csv")
x = data['x'].values
y = data['y'].values

# 构建概率模型
with pm.Model() as model:
    # 参数的先验分布
    theta1 = pm.Lognormal('theta1', mu=0, sigma=10)
    theta3 = pm.Lognormal('theta3', mu=0, sigma=10)
    theta4 = pm.Lognormal('theta4', mu=0, sigma=10)
    theta2 = pm.Uniform('theta2', lower=0, upper=1)

    # 使用 erfc 来实现Φ(-x/θ4)
    # Φ(-x/θ4) = 0.5 * erfc(x/(θ4*sqrt(2)))
    norm_cdf = 0.5 * pm.math.erfc(x / (theta4 * pm.math.sqrt(2)))

    # 定义均值函数
    mu = (
        theta1 * pm.math.exp(-theta2 * x**2) +
        theta3 * (1 - pm.math.exp(-theta2 * x**2)) * norm_cdf
    )

    # 观测数据的似然函数
    y_obs = pm.Poisson('y_obs', mu=mu, observed=y)

    # 采样设置：10,000个后验样本，1,000个调整样本
    trace = pm.sample(10000, tune=1000, target_accept=0.95, random_seed=42)

# 后验样本的统计摘要
summary = az.summary(trace, hdi_prob=0.95)
print(summary)

# 使用后验分布预测厚度为 1.5 nm 时的纳米线密度
x_new = 1.5
theta1_samples = trace.posterior['theta1'].values.flatten()
theta3_samples = trace.posterior['theta3'].values.flatten()
theta4_samples = trace.posterior['theta4'].values.flatten()
theta2_samples = trace.posterior['theta2'].values.flatten()

# 计算新 x 值下的 Φ(-x_new/θ4)
norm_cdf_new = norm.cdf(-x_new / theta4_samples)

mu_new = (
    theta1_samples * np.exp(-theta2_samples * x_new**2) +
    theta3_samples * (1 - np.exp(-theta2_samples * x_new**2)) * norm_cdf_new
)

# 预测分布
predictive_samples = np.random.poisson(mu_new)
predictive_mean = np.mean(predictive_samples)
predictive_hdi = az.hdi(predictive_samples, hdi_prob=0.95)

# 输出预测结果
print(f"Predictive mean: {predictive_mean}")
print(f"95% HDI: {predictive_hdi}")

