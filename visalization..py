# Heatmap
data_corr = data[numeric_columns].corr()
sns.heatmap(data_corr, annot=True, cmap='coolwarm')
plt.ylabel('')
plt.show()

# 2 variables
group1 = data.groupby('Churn')['Avg_Workout_Duration_Min'].mean().sort_values(ascending=False)
plt.figure(figsize=(5,5))
plt.xlabel('Average Duration in Mins')
plt.ylabel('Churn')
bars = plt.bar(group1.index, group1.values, color='blue', edgecolor='black')
plt.show()

# 3 variables
data.groupby(['Churn', 'Membership_Type'])['Total_Weight_Lifted_kg'].mean().unstack().plot(kind='bar')
plt.title('Average weight lifted by Churn status')
plt.xlabel('Churn')
plt.ylabel('Average weight in kg')
plt.legend(title='Membership Type')
plt.show()