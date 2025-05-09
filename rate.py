import matplotlib.pyplot as plt
import numpy as np

# Data: Organization types and their corresponding AI adoption rates
organization_types = ['Independent PCPs', 'Group Practices', 'Integrated Health Systems',
                      'Rural Practices', 'Urban Practices', 'Large Hospitals', 'Community Health Centers']
adoption_rates = [55, 72, 80, 50, 75, 85, 60]

# Calculate mean and standard deviation
mean_adoption_rate = np.mean(adoption_rates)
std_dev_adoption_rate = np.std(adoption_rates)

# Display the results
print(f"Mean AI Adoption Rate: {mean_adoption_rate:.2f}%")
print(f"Standard Deviation of AI Adoption Rates: {std_dev_adoption_rate:.2f}%")

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(organization_types, adoption_rates, color='skyblue')
plt.xlabel('Organization Type')
plt.ylabel('AI Adoption Rate (%)')
plt.title('AI Adoption Rates Among PCPs by Organization Type')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
