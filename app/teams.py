valid_teams=[

    'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',

    'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich Town',

    'Leicester City', 'Liverpool', 'Manchester City', 'Manchester Utd',

    'Newcastle United', "Nottingham Forest", 'Southampton', 'Tottenham',

    'West Ham United', 'Wolves', 'Atalanta', 'Bologna', 'Cagliari', 'Como',

    'Empoli', 'Fiorentina', 'Genoa', 'Hellas Verona', 'Inter',

    'Juventus', 'Lazio', 'Lecce', 'Milan', 'Monza', 'Napoli', 'Parma',

    'Roma', 'Torino', 'Udinese', 'Venezia', 'Augsburg',

    'Bayern Munich', 'Bochum', 'Dortmund', 'Eint Frankfurt',

    'Freiburg', 'Gladbach', 'Heidenheim', 'Hoffenheim',

    'Holstein Kiel', 'Leverkusen', 'Mainz 05', 'RB Leipzig',

    'St. Pauli', 'Stuttgart', 'Union Berlin', 'Werder Bremen',

    'Wolfsburg', 'Angers', 'Auxerre', 'Brest', 'Le Havre', 'Lens',

    'Lille', 'Lyon', 'Marseille', 'Monaco', 'Montpellier', 'Nantes',

    'Nice', 'Paris Saint-Germain', 'Reims', 'Rennes', 'Saint-Étienne',

    'Strasbourg', 'Toulouse', 'Alavés', 'Athletic Club',

    'Atlético Madrid', 'Barcelona', 'Betis', 'Celta Vigo', 'Espanyol',

    'Getafe', 'Girona', 'Las Palmas', 'Leganés', 'Mallorca', 'Osasuna',

    'Rayo Vallecano', 'Real Madrid', 'Real Sociedad', 'Sevilla',

    'Valencia', 'Valladolid', 'Villarreal', 'Blackburn',

    'Bristol City', 'Burnley', 'Cardiff City', 'Coventry City',

    'Derby County', 'Hull City', 'Leeds United', 'Luton Town',

    'Middlesbrough', 'Millwall', 'Norwich City', 'Oxford United',

    'Plymouth Argyle', 'Portsmouth', 'Preston', 'QPR', 'Sheffield Utd',

    'Sheffield Weds', 'Stoke City', 'Sunderland', 'Swansea City',

    'Watford', 'West Brom', 'Bari', 'Brescia', 'Carrarese',

    'Catanzaro', 'Cesena', 'Cittadella', 'Cosenza', 'Cremonese',

    'Frosinone', 'Juve Stabia', 'Mantova', 'Modena', 'Palermo', 'Pisa',

    'Reggiana', 'Salernitana', 'Sampdoria', 'Sassuolo', 'Spezia',

    'Südtirol'

]


"""# Optional: show bar chart for top 5
        st.markdown("### 📊 Top 5 Predictions by Probability Gap")
        for idx, row in df_predictions.head(5).iterrows():
            st.markdown(f"**{row['Home_Team']} vs {row['Away_Team']}**")
            fig, ax = plt.subplots()
            ax.bar(
                ['Away Win', 'Draw', 'Home Win'],
                [row['Away_Win_Prob'], row['Draw_Prob'], row['Home_Win_Prob']],
                color=['#ff4b4b', '#ffa500', '#4CAF50']
            )
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_title("Outcome Probabilities")
            for i, val in enumerate([row['Away_Win_Prob'], row['Draw_Prob'], row['Home_Win_Prob']]):
                ax.text(i, val + 0.02, f"{val:.1%}", ha='center')
            st.pyplot(fig)"""