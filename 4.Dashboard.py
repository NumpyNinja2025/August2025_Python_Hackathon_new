import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("GDM_Python_Aug2025.xlsx")
df = data.copy()

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
st.set_page_config(layout="wide")

df['Height_m'] = df['Height_cms'] / 100
df['BMI'] = df['WeightinV1'] / (df['Height_m'] ** 2)
df['BMI'] = df['BMI'].round(1)

# Categorize BMI
def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

df['BMI_Category'] = df['BMI'].apply(bmi_category)

# Map Age and GDM
df['Age_Group'] = df['Age_gt_30'].map({'No': '<=30', 'Yes': '>30'})
df['GDM_Status'] = df['GDM Diagonised'].map({'No': 'No GDM', 'Yes': 'GDM'})

# --- Streamlit UI ---

st.title("BMI & GDM Analysis Dashboard")
st.title("Descriptive Analysis")
# Layout: Side-by-side columns
col1, col2,col3 = st.columns(3)

with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.histplot(df['BMI'], bins=20, kde=True, ax=ax1)
    ax1.set_title('Distribution of BMI')
    ax1.set_xlabel('BMI')
    ax1.set_ylabel('Frequency')
    st.pyplot(fig1)
    st.write("Higher BMI may be correlated to a higher risk of GDM. We also found higher resting heart rates and higher risk of emergency C-sections among those with BMI in this dataset.75%: Are below 29.9, but many are close to the obesity threshold.")


with col2:
    counts = pd.crosstab(df['GDM_Status'], df['Age_Group'])
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    counts.plot(kind='bar', ax=ax2)
    ax2.set_title('GDM Status by Age Group')
    ax2.set_xlabel('GDM Status')
    ax2.set_ylabel('Count')
    ax2.legend(title='Age Group')
    plt.xticks(rotation=0)
    st.pyplot(fig2)
    st.write("Those above 30 seem to be at a higher risk of GDM within our dataset, with 79% of those with GDM >30 years of age. So this is an important statistic to show")


with col3:  

    # Compute means
    gdm = df[df['GDM Diagonised'] == 'Yes']['BirthWeight'].mean()
    no_gdm = df[df['GDM Diagonised'] == 'No']['BirthWeight'].mean()
    overall = df['BirthWeight'].mean()

    # Pie chart data
    labels = ['With GDM', 'Without GDM']
    values = [gdm, no_gdm]
    colors = ['#FF9999', '#99CCFF']

    # Custom labels with actual values
    def value_labels(pct, allvals):
        absolute = int(round(pct/100. * sum(allvals)))
        return f"{absolute} g"

    # Plot
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct=lambda pct: f"{pct/100. * sum(values):.3f} g",
        colors=colors,
        startangle=90,
        textprops=dict(color="black", fontsize=10)
    )
    ax.axis('equal')
    plt.title("Avg Birth Weight by GDM Status")

    st.pyplot(fig)
    st.write("Babies born to mothers with GDM tend to weigh slightly less on average than those without GDM — a 0.12 kg difference. This might reflect tighter glucose control, earlier delivery, or medical interventions.")

st.title("Multivariate Analysis")

col4, col5,col6 = st.columns(3)

with col4:  

    # Use median of ALT_v1 and ALT_v2
    df['Elevated_ALT'] = ((df['ALT_V1'] + df['ALT_V3']) / 2) > 35

    # Calculate % with elevated ALT in each GDM category
    alt_gdm_counts = df.groupby('GDM Diagonised')['Elevated_ALT'].value_counts(normalize=True).unstack() * 100
    non_gdm_elevated = alt_gdm_counts.loc['No', True] if 'No' in alt_gdm_counts.index else 0
    gdm_elevated = alt_gdm_counts.loc['Yes', True] if 'Yes' in alt_gdm_counts.index else 0

# Step 4: Display results
    # Extract values for plotting
    elevated_percent = alt_gdm_counts[True].reindex(['No', 'Yes'])  # Order: Non-GDM, GDM
    elevated_percent.index = ['Non-GDM', 'GDM']

    # Plot horizontal bar chart
    fig, ax = plt.subplots(figsize=(6, 3))
    elevated_percent.plot(kind='barh', ax=ax, color=['#99CCFF', '#FF9999'])
    ax.set_xlabel('% with Elevated ALT')
    ax.set_xlim(0, max(elevated_percent.max() + 5, 20))
    ax.set_title('Elevated ALT Prevalence by GDM Status')
    for i, v in enumerate(elevated_percent):
        ax.text(v + 0.5, i, f"{v:.1f}%", va='center')
    st.pyplot(fig)    
    st.write(f"📊 **% with Elevated ALT (>35)**")
    st.write(f"- **Non-GDM:** {non_gdm_elevated:.2f}%")
    st.write(f"- **GDM:** {gdm_elevated:.2f}% (≈ {gdm_elevated/non_gdm_elevated:.1f}× higher than Non-GDM)" if non_gdm_elevated > 0 else "-")
    st.write("Babies born to mothers with GDM tend to weigh slightly less on average than those without GDM — a 0.12 kg difference. This might reflect tighter glucose control, earlier delivery, or medical interventions.")

with col5:

    # Step 1: Clean the data
    df_clean = df[(df['GDM Diagonised'] != 'NR') & (df['Caesarean'] != 'NR')]

    # Step 2: Calculate proportions
    proportions = (
        df_clean.groupby('GDM Diagonised')['Caesarean']
        .value_counts(normalize=True)
        .unstack()
        .fillna(0) * 100
    ).round(1)

    # Step 3: Prepare plot data
    plot_data = pd.DataFrame({
        'GDM Diagonised': proportions.index,
        'Cesarean (%)': proportions[1],
        'Vaginal (%)': proportions[0]
    })

    # Step 4: Plot in Streamlit
    fig, ax = plt.subplots(figsize=(7, 5))
    bar_width = 0.4
    x = range(len(plot_data))

    # Bars
    ax.bar(x, plot_data['Cesarean (%)'], width=bar_width, label='Cesarean', color='salmon')
    ax.bar([i + bar_width for i in x], plot_data['Vaginal (%)'], width=bar_width, label='Vaginal', color='lightblue')

    # X-axis and labels
    ax.set_xticks([i + bar_width / 2 for i in x])
    ax.set_xticklabels(plot_data['GDM Diagonised'])
    ax.set_ylabel('Delivery Method Proportion (%)')
    ax.set_title('Delivery Method Proportions by GDM Status')
    ax.legend()

    # Annotate percentages
    for i in x:
        ax.text(i, plot_data['Cesarean (%)'].iloc[i] + 1,
                f"{plot_data['Cesarean (%)'].iloc[i]}%", ha='center')
        ax.text(i + bar_width, plot_data['Vaginal (%)'].iloc[i] + 1,
                f"{plot_data['Vaginal (%)'].iloc[i]}%", ha='center')

    st.pyplot(fig)
    st.write("The Cesarean rate is notably higher among those diagnosed with GDM—47.3% vs 31.2%—suggesting that GDM may be associated with increased likelihood of Cesarean delivery. This could reflect medical decisions based on fetal or maternal risk factors tied to GDM (like macrosomia or labor complications).")
    
with col6:
    # Step 1: Classify anemia
    def classify_anemia(hb_value):
        if hb_value < 8.0:
            return "Severe Anemia"
        elif 8.0 <= hb_value <= 10.9:
            return "Moderate Anemia"
        elif 11.0 <= hb_value <= 11.9:
            return "Mild Anemia"
        else:
            return "Normal"

    df['Anemia_Category'] = df['Hemoglobin_V1'].apply(classify_anemia)

    # Step 2: Ensure Caesarean is numeric
    df['Caesarean'] = pd.to_numeric(df['Caesarean'], errors='coerce')

    # Step 3: Drop missing values
    df_clean = df.dropna(subset=['Caesarean', 'Anemia_Category', 'GDM Diagonised'])

    # Step 4: Calculate Caesarean rate by Anemia Category and GDM
    grouped = df_clean.groupby(['Anemia_Category', 'GDM Diagonised'])['Caesarean'].mean().reset_index()
    grouped['Caesarean'] *= 100

    # Step 5: Define desired anemia category order
    anemia_order = ["Severe Anemia", "Moderate Anemia", "Mild Anemia", "Normal"]
    grouped['Anemia_Category'] = pd.Categorical(grouped['Anemia_Category'], categories=anemia_order, ordered=True)
    grouped = grouped.sort_values(['Anemia_Category', 'GDM Diagonised'])

    # Step 6: Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=grouped,
        x='Anemia_Category',
        y='Caesarean',
        hue='GDM Diagonised',
        palette='Set2',
        ax=ax
    )

    # Add labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', label_type='edge', padding=2)

    ax.set_ylabel("Caesarean Rate (%)")
    ax.set_title("Caesarean Rate by Anemia Severity and GDM Status")
    ax.legend(title='GDM Status')
    plt.xticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig)
    st.write("The Caesarean rates were 33.3% for those with moderate anemia, 58.3% for mild anemia, and 46.6% for those with normal hemoglobin.These findings suggest that even mild reductions in hemoglobin, when combined with GDM, may significantly elevate the risk of Caesarean deliveries.")


    
    


