import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

df = pd.read_csv(r"C:\Users\Al kubaisi Mehdi\Desktop\DATASCIENTEST\Projetnov23DA\atp_data_OR.csv")

st.sidebar.title("Sommaire")
pages = ["Projet","Exploration des données", "Visualisation des données", "Machine Learning","Résultats","Simulation", "Les auteurs"]
page = st.sidebar.radio("Aller vers", pages)
st.sidebar.header("Auteurs")
st.sidebar.markdown("[**Al kubaisi Mehdi**](https://www.linkedin.com/in/mehdialkubaisi/)")
st.sidebar.markdown("[**Achour Amar**](https://www.linkedin.com/in/amar-achour-765804124/)")
st.sidebar.markdown("[**Chicheportiche Jonathan**](https://www.waza.org/404)")


if page == pages[0]:
    st.write("# Projet Paris Sportifs")
    st.write("## Prédire les gagnants de matchs de tennis : Un défi de datascience")
    st.write("Le monde du sport regorge d'opportunités pour l'analyse de données et l'apprentissage automatique. Le tennis, avec ses statistiques précises et ses matchs dynamiques, offre un terrain de jeu fertile pour explorer le potentiel de ces approches. Ce projet vise à relever un défi passionnant : développer un modèle de machine learning capable de surpasser les algorithmes des bookmakers dans la prédiction des gagnants de matchs de tennis.")
    st.write("####    Déroulement du projet :")
    st.write("Le projet se déroulera en plusieurs étapes clés, suivant la méthodologie rigoureuse de la science des données. Nous allons pour cela traiter l'ensemble des étapes d'un projet de Data Science:")
   
    st.write("#####     1. Exploration et nettoyage des données :")
    st.write( """- Plongeons dans l'ensemble de données pour en comprendre la structure, les caractéristiques et les éventuelles anomalies.\n
- Nettoyage minutieux des données pour garantir leur qualité et leur fiabilité.\n
- Visualisation des données pour identifier des tendances et des modèles pertinents. """)
    
    st.write("#####     2. Randomisation des données :")
    st.write("""- Identification des variables fortement corrélées avec le résultat du match.\n
- Mise en place d'une stratégie de randomisation pour neutraliser l'impact de ces variables et garantir l'impartialité du modèle.""")
    
    st.write("#####     3. Feature engineering :")
    st.write("""- Création de nouvelles variables à partir des données existantes, augmentant ainsi la richesse de l'information disponible.\n
- Création de la variable cible qui devra prédire le gagnant d'un match donné.
- Sélection des caractéristiques les plus pertinentes pour la prédiction des résultats.""")
    
    st.write("#####     4. Entraînement et évaluation du modèle:")
    st.write("""- Développement de différents modèles d'apprentissage automatique pour prédire les gagnants de matchs.\n
- Évaluation rigoureuse des performances des modèles. """)
    
    st.write("#####     5. Optimisation et déploiement du modèle :")
    st.write("""- Affinement du modèle sélectionné pour maximiser sa précision et sa robustesse.\n
- Déploiement du modèle en production pour une utilisation réelle dans la prédiction des résultats de matchs de tennis.""")
    
    st.write("""Ce projet de datascience promet d'être une aventure stimulante, combinant l'analyse de données  
             et l'application pratique de l'apprentissage automatique. En relevant plusieurs défis dont celui de randomiser les données et de surpasser les bookmakers, 
             nous explorerons le potentiel de la science des données pour réussir à battre les bookmakers !""")
    
    st.image("tennis_image.jpg" )
    
elif page == pages[1]:
    st.write("# Exploration et nettoyage des données")
    st.write("##### Premier regard")
    st.write(df.head())
    st.write("Les dimensions du datasets sont:", df.shape)
    st.write("###### Que représentent les variables de ce dataset ? ")
    st.write("Ce Dataset représente 44708 différents matchs de tennis hommes joués entre janvier 2000 et Mars 2018. Nous pouvons décrire les 23 variables le composant de la manière suivante :")
    st.write("""- **ATP**: Chaque numéro correspond à un type de tournoi ( Adelaide = 1, Doha = 3)
- **Location** : Lieu du tournoi
- **Tournament** : Nom du tournoi
- **Date**: Date du match joué
- **Serie** : Correspond à un type de tournoi (Grand Slam, International, ATP250)
- **Court** : Correspond à un terrain en intérieur ou extérieur
- **Surface** : Correspond au type de terrain (gazon, terre battue..)
- **Round** : Niveau de la compétition
- **Best of** : Nombre de sets à remporter pour gagner le match
- **Winner / Looser** : Gagnant / perdant du match
- **Wrank / Lrank** : Correspond au classement mondial du vainqueur/perdant du match
- **Wsets / Lsets** : Nombre de sets remportés par le vainqueur / perdant
- **Comment** : Correspond à la manière dont le match s'est terminé ('Completed' 'Retired' 'Walkover' ou 'Disqualified')
- **PSW / PSL** : Cote du bookmaker Pinnacle gagnant / perdant
- **B365W / BSWL** : Cote du bookmaker B365 gagnant / perdant
- **Elo_winner / elo_looser** : elo du gagnant / perdant
- **Proba_elo** : probabilité du vainqueur en fonction de la différence de elo des joueurs""")
    
    st.write("###### Où se trouve les valeurs manquantes de notre jeu de données ? ")
    def get_data_summary(df):
        summary_df = pd.DataFrame(index=df.columns, columns=[ "Unique Values", "Missing Values", "Data Type", "% Missing"])

        summary_df["Unique Values"] = df.nunique()
        summary_df["Missing Values"] = df.isna().sum()
        summary_df["Data Type"] = df.dtypes
        summary_df["% Missing"] = round((df.isna().sum() / len(df)) * 100, 2)

        return summary_df

    if st.checkbox("Show Missing Values Summary"):
        summary_df = get_data_summary(df)
        st.dataframe(summary_df.astype(str))
        
    st.write("###### Regardons les valeurs des variables numeriques et leurs distributions ")
    st.write(round(df.describe(),2))
    st.write("###### Que pouvons-nous dire de ces variables ?")
    st.write("""Inspectons ces variables avec la méthode describe, info et notre tableau récapitulant les valeurs manquantes :\n

Après inspection des différentes modalités de chaque variable qualitative il ne semble pas y avoir d'erreur dans celles-ci.
Aucune valeur manquante n'est présente dans les variables qualitatives.\n
En observant les variables quantitatives nous pouvons ressortir quelques informations :\n

- Le rang mondial le plus faible est 2000.
- La plus grosse cote pour un gagnant chez Pinnacle est de 46, la plus grosse cote pour un perdant est de 121.
- Chez B365 la plus grosse cote pour un gagnant est de 29 et la plus grosse cote pour un perdant est de 101.
- La cote moyenne pour un vainqueur chez Pinnacle est de 1.927, elle est de 1.822 chez B365.
- La cote moyenne pour un perdant chez Pinnacle est de 4.24, elle est de 3.55 chez B365.
- Les variables PSW, PSL, B365W, B365L sont celles possédant le plus de valeurs manquantes avec près de 27 % chez Pinnacle et 13 % chez B365.
- Le elo moyen des vainqueurs est de 1684, celui des perdants est de 1609.\n
Donc, d'après ces premières observations nous pouvons déjà constater que le bookmaker Pinnacle propose des cotes plus généreuses sur les matchs ATP. Concernant les valeurs manquantes il faudra vérifier la raison; une première hypothèse est que ce dataset commence avec des matchs de tennis joués en 2000 et que les bookmakers ne proposaient peut-être pas tous les matchs dans leur paris à cette époque. Nous pourrons vérifier facilement cela avec un graphique répertoriant le nombre de valeurs manquantes au fil des années pour les variables PSW, PSL, B365W et B365L.""")
    
    options = ["Court","Surface","Location","Series","Round","Comment","Best of"]
    selected_variable = st.selectbox("Variables catégoriels", options)
    if selected_variable == "Court":
        st.write("Les différentes modalités de 'Court' sont:", df["Court"].unique())
    elif selected_variable == "Surface":
        st.write("Les différentes modalités de 'Surface' sont:", df["Surface"].unique())
    elif selected_variable == "Location":
        st.write("Les différentes modalités de 'Location' sont:", df["Location"].unique())
    elif selected_variable == "Series":
        st.write("Les différentes modalités de 'Series' sont:", df["Series"].unique())
    elif selected_variable == "Round":
        st.write("Les différentes modalités de 'Round' sont:", df["Round"].unique())
    elif selected_variable == "Comment":
        st.write("Les différentes modalités de 'Comment' sont:", df["Comment"].unique())
    elif selected_variable == "Best of":
        st.write("Les différentes modalités de 'Best of' sont:", df["Best of"].unique())

elif page == pages[2]:
    st.write("# Visualisation des données")
    st.write("##### 1. Visualisation des valeurs manquantes")
    st.write("Nous avons vu précédement qu'un grand nombre de valeurs manquantes étaient présente sur les variables contenant les cotes des bookmakers. Un graphique pour visualiser cela pourrait nous aider.")
     
    df["Date"] = pd.to_datetime(df["Date"])
    df['Year'] = df['Date'].dt.year

    # Créer de nouvelles colonnes consolidant les valeurs manquantes pour PSW et PSL, B365W et B365L
    df['PS'] = df['PSW'].combine_first(df['PSL'])
    df['B365'] = df['B365W'].combine_first(df['B365L'])

    # Créer un DataFrame avec le nombre de valeurs manquantes consolidées par année pour chaque bookmaker
    missing_data_by_year = df.groupby('Year')[['PS', 'B365']].apply(lambda x: x.isnull().sum()).reset_index()

    # Transformer les données en format long avec pd.melt afin
    missing_data_long = pd.melt(missing_data_by_year, id_vars=['Year'], var_name='Bookmaker', value_name='Missing Values')   
    fig = px.bar(missing_data_long, x= "Year",
             y = "Missing Values",
             color = "Bookmaker", barmode = "group",
             color_discrete_sequence=['#2E8B57', '#F7DC6F', '#FFFFFF','#D2691E',"#2E8B57"],
             title = "<b>Nombre de valeurs manquantes par année<b>")
    fig.update_layout(yaxis_title='<b>Valeurs Manquantes</b>',
    xaxis_title="<b>Années<b>",
    plot_bgcolor='white',
    paper_bgcolor='white',
    width=800, height=500)
    st.plotly_chart(fig)
    
    st.write(""" **Observations** :  
             
- En 2000 nous ne possédons aucune valeur pour les cotes des matchs de tennis.  
- En 2001 il y a 93 valeurs manquantes chez les deux bookmakers.  
- En 2002 et 2003 nous ne possédons aucune donnée concernant les cotes de Pinnacle.  
- Les années suivantes ne présentent pas de valeurs abberantes concernant les données manquantes sauf en 2009 où nous ne possédons aucune valeur de cote pour Pinnacle cette année là.  
- Plus les années sont récentes, plus les données manquantes sont rares.""")
     
    df = df.dropna(subset=['Wsets'])
    df = df.dropna(subset=['Lsets'])
    diff_365 = df.loc[(df["B365W"].isna()) & (df["B365L"].notna())]
    df = df.drop(diff_365.index)
    
    st.write(" **Les différents courts de tennis, types de terrain et comments**")
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))  # 1 ligne, 3 colonnes

    sns.histplot(df["Court"], ax=ax[0], color = '#2E8B57')
    ax[0].set_title('Interieur / Extérieur')
    labels1 = ["Intérieur","Extérieur"]
    ax[0].set_xticks(ax[0].get_xticks())
    ax[0].set_xticklabels(labels1, rotation=45)

    sns.histplot(df["Comment"], ax=ax[1], color = '#2E8B57')
    labels2 = ["Complété", "Forfait", "Walkover", "Disqualifié"]
    ax[1].set_title('Fin de match')
    ax[1].set_xticks(ax[1].get_xticks())
    ax[1].set_xticklabels(labels2, rotation=45)

    sns.histplot(df["Surface"], ax=ax[2],color = '#2E8B57')
    labels3 = ["Dur","Terre battue","Herbe","Synthétique"]
    ax[2].set_title('Surfaces')
    ax[2].set_xticks(ax[2].get_xticks())
    ax[2].set_xticklabels(labels3, rotation=45)

    plt.subplots_adjust(wspace=0.3)
    st.pyplot(fig)
    
    st.write("Visualisation de la distribution des cotes en fonction des variables")
    options = ["Bookmakers","Nombre de sets", "Type de tournois","Type de terrains","Nombre de rounds"]
    selected_variable = st.selectbox("Distributions des cotes des bookmakers", options)
    if selected_variable == "Bookmakers":
        palette = tennis_colors = ["#009900","#FFAB00","#ADFF2F",'#F7DC6F']
        fig, ax = plt.subplots(figsize =(8,5))  # Créer une figure et un objet Axes
        sns.boxplot(data=df[['PSW',"B365W",'PSL',"B365L"]],showfliers=False, orient = "h", palette = palette)
        plt.title('Distribution des cotes de bookmaker', fontsize=16)
        plt.xlabel('Cotes', fontsize=14)
        plt.xticks(np.arange(1, 9, step=0.5))
        st.pyplot(fig)
    
    elif selected_variable == "Nombre de sets":
        fig, axes = plt.subplots(4, 1, figsize=(6, 15))
        palette = ['#2E8B57', '#F7DC6F']

        # Graphique numéro 1 permettant de voir la distribution de B365W par séries
        sns.boxplot(data=df, x="Best of", y="B365W", ax=axes[0], showfliers=False, hue="Best of", palette=palette)
        axes[0].set_title("Distribution des cotes B365W par nombre de sets")
        axes[0].set_xlabel("Sets")
        axes[0].set_ylabel("Cotes B365W")
        axes[0].set_yticks(np.arange(1, 4, step=0.5))

        # Graphique numéro 2 permettant de voir la distribution de B365L par nombre de sets
        sns.boxplot(data=df, x="Best of", y="B365L", ax=axes[1], showfliers=False, hue="Best of", palette=palette)
        axes[1].set_title("Distribution des cotes B365L par nombre de sets")
        axes[1].set_xlabel("Sets")
        axes[1].set_ylabel("Cotes B365L")
        axes[1].set_yticks(np.arange(1, 15, step=1))

        # Graphique numéro 3 permettant de voir la distribution de PSW par nombre de sets
        sns.boxplot(data=df, x="Best of", y="PSW", ax=axes[2], showfliers=False, hue="Best of", palette=palette)
        axes[2].set_title("Distribution des cotes PSW par nombre de sets")
        axes[2].set_xlabel("Sets")
        axes[2].set_ylabel("Cotes PSW")
        axes[2].set_yticks(np.arange(1, 5, step=0.5))

        # Graphique numéro 4 présentant la distribution de PSL par nombre de sets
        sns.boxplot(data=df, x="Best of", y="PSL", ax=axes[3], showfliers=False, hue="Best of", palette=palette)
        axes[3].set_title("Distribution des cotes PSL par nombre de sets")
        axes[3].set_xlabel("Sets")
        axes[3].set_ylabel("Cotes PSL")
        axes[3].set_yticks(np.arange(1, 15, step=1))

        plt.tight_layout()
        st.pyplot(fig)
        
    elif selected_variable == "Type de tournois":
        fig, axes = plt.subplots(figsize= (15,8))
        palette = [ "#009900", '#2E8B57',"#ADFF2F",'#F7DC6F', "#FFD700", "#FFAB00", "#F39C12", "#E67E22"]
        #Graphique numéro 1 permettant de voir la distribution de B365 par type de tournois
        sns.boxplot(data=df, x="Series", y="B365W", ax=axes,showfliers=False, palette =palette)
        axes.set_title("Distribution des cotes B365W par type de tournois", size = 16)
        axes.set_xlabel("Tournois")
        axes.set_ylabel("Cotes B365W")
        axes.set_yticks(np.arange(1, 4, step=0.5));
        axes.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

        #Graphique numéro 2 permettant de voir la distribution de B365L par type de tournois
        fig, axes = plt.subplots(figsize= (15,8))
        sns.boxplot(data=df, x="Series", y="B365L", ax=axes,showfliers=False,palette =palette)
        axes.set_title("Distribution des cotes B365l par type de tournois",size = 16)
        axes.set_xlabel("Tournois")
        axes.set_ylabel("Cotes B365l")
        axes.set_yticks(np.arange(1, 10, step=0.5));
        axes.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        #Graphique numéro 3 permettant de voir la distribution de PSW par type de tournois
        
        fig, axes = plt.subplots(figsize= (15,8))
        sns.boxplot(data=df, x="Series", y="PSW", ax=axes,showfliers=False,palette =palette)
        axes.set_title("Distribution des cotes PSW par type de tournois",size = 16)
        axes.set_xlabel("Tournois")
        axes.set_ylabel("Cotes PSW")
        axes.set_yticks(np.arange(1, 5, step=0.5));
        axes.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        #Graphique numéro 4 présentant la distribution de PSL par type de tournois
        
        fig, axes = plt.subplots(figsize= (15,8))
        sns.boxplot(data=df, x="Series", y="PSL", ax=axes,showfliers=False,palette =palette)
        axes.set_title("Distribution des cotes PSl par type de tournois",size = 16)
        axes.set_xlabel("Tournois")
        axes.set_ylabel("Cotes PSL")
        axes.set_yticks(np.arange(1, 13, step=0.5))
        axes.tick_params(axis='x', rotation=45)
        plt.tight_layout();
        st.pyplot(fig)
        
    elif selected_variable =="Type de terrains":
        fig, axes = plt.subplots(figsize= (15,8))
        palette = ["#ADFF2F",'#F7DC6F', "#FFD700", "#FFAB00"]

        #Graphque numéro 1 permettant de voir la distribution de B365 par types de terrain
        sns.boxplot(data=df, y="B365W",x="Surface", ax=axes,showfliers=False, palette =palette)
        axes.set_title("Distribution des cotes B365W par types de terrain", size = 16)
        axes.set_xlabel("Terrain")
        axes.set_ylabel("Cotes B365W")
        axes.set_yticks(np.arange(1, 4, step=0.5));
        st.pyplot(fig)

        #Graphique numéro 2 permettant de voir la distribution de B365L par types de terrain
        fig, axes = plt.subplots(figsize= (15,8))
        sns.boxplot(data=df, x="Surface", y="B365L", ax=axes,showfliers=False, palette =palette)
        axes.set_title("Distribution des cotes B365l par types de terrain", size = 16)
        axes.set_xlabel("Terrain")
        axes.set_ylabel("Cotes B365l")
        axes.set_yticks(np.arange(1, 10, step=0.5));
        st.pyplot(fig)

        #Graphique numéro 3 permettant de voir la distribution de PSW par types de terrain
        fig, axes = plt.subplots(figsize= (15,8))

        sns.boxplot(data=df, x="Surface", y="PSW", ax=axes,showfliers=False, palette =palette)
        axes.set_title("Distribution des cotes PSW par types de terrain", size = 16)
        axes.set_xlabel("Terrain")
        axes.set_ylabel("Cotes PSW")
        axes.set_yticks(np.arange(1, 5, step=0.5));
        st.pyplot(fig)

        #Graphique numéro 4 présentant la distribution de PSL par types de terrain       
        fig, axes = plt.subplots(figsize= (15,8))
        sns.boxplot(data=df, x="Surface", y="PSL", ax=axes,showfliers=False, palette =palette)
        axes.set_title("Distribution des cotes PSl par types de terrain", size = 16)
        axes.set_xlabel("Terrain")
        axes.set_ylabel("Cotes PSL")
        axes.set_yticks(np.arange(1, 13, step=0.5))
        plt.tight_layout();
        st.pyplot(fig)
        
    elif selected_variable =="Nombre de rounds": 
        fig, axes = plt.subplots(figsize= (15,8))
        palette = [ "#009900", '#2E8B57',"#ADFF2F",'#F7DC6F', "#FFD700", "#FFAB00", "#F39C12", "#E67E22"]

        #Graphque numéro 1 permettant de voir la distribution de B365 par round
        sns.boxplot(data=df, x="Round", y="B365W", ax=axes,showfliers=False, palette =palette)
        axes.set_title("Distribution des cotes B365W par round", size = 16)
        axes.set_xlabel("Rounds")
        axes.set_ylabel("Cotes B365W")
        axes.set_yticks(np.arange(1, 4, step=0.5));
        axes.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

        #Graphique numéro 2 permettant de voir la distribution de B365L par round
        fig, axes = plt.subplots(figsize= (15,8))
        sns.boxplot(data=df, x="Round", y="B365L", ax=axes,showfliers=False, palette =palette)
        axes.set_title("Distribution des cotes B365l par round", size = 16)
        axes.set_xlabel("Rounds")
        axes.set_ylabel("Cotes B365l")
        axes.set_yticks(np.arange(1, 10, step=0.5));
        axes.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        #Graphique numéro 3 permettant de voir la distribution de PSW par type de tournoi
        fig, axes = plt.subplots(figsize= (15,8))
        sns.boxplot(data=df, x="Round", y="PSW", ax=axes,showfliers=False, palette =palette)
        axes.set_title("Distribution des cotes PSW par type de tournoi", size = 16)
        axes.set_xlabel("Rounds")
        axes.set_ylabel("Cotes PSW")
        axes.set_yticks(np.arange(1, 5, step=0.5));
        axes.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        #Graphique numéro 4 présentant la distribution de PSL par type de tournoi
        fig, axes = plt.subplots(figsize= (15,8))
        sns.boxplot(data=df, x="Round", y="PSL", ax=axes,showfliers=False, palette =palette)
        axes.set_title("Distribution des cotes PSl par type de tournoi", size = 16)
        axes.set_xlabel("Rounds")
        axes.set_ylabel("Cotes PSL")
        axes.set_yticks(np.arange(1, 13, step=0.5))
        axes.tick_params(axis='x', rotation=45)
        plt.tight_layout();
        st.pyplot(fig)
                
    st.write("**Valeur des cotes en fonction de la différence de elo**")   
    plt.figure(figsize=(12,6))
    fig, ax = plt.subplots(figsize= (15,8))
    ax=sns.scatterplot(x='proba_elo', y='PSW', data=df, alpha =0.5, color = "#1fa187", label = "Cote Gagnante")
    ax2 = plt.twinx()
    ax1=sns.scatterplot(x='proba_elo', y='PSL', data=df,  alpha = 0.8, color = "#365c8d",ax = ax2,label = "Cote perdante ")
    sns.move_legend(ax1, "upper left", bbox_to_anchor=(0.10, 1))
    plt.title('Valeur de la cote en fonction de la différence de elo', size = 16)
    ax2.set_ylabel('PSL')
    ax.set_xlabel("Différence de ELO")
    plt.ylim(ymax=60, ymin = 0)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(0.10, 0.94))
    st.pyplot(fig)   
    st.write(""" Ce dernier graphique semble intéressant. Nous utilisons la variable porba_elo qui représente la différence de elo entre les deux joueurs. Lorsque la valeur est de 0.5 alors leur elo est égal. Plus on se rapproche de 0 ou de 1, plus la différence est grande. Grâce à ce graphique nous pouvons voir que plus la différence de elo est grande plus les cotes Losers sont hautes et que les cotes Winners sont grandes.""")
    
    st.write("**Analyse des corrélations**") 
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize = (10,6))
    df_num = df[[ 'Best of', 'WRank', 'LRank', 'Wsets','Lsets', 'PSW', 'PSL', 'B365W', 'B365L', 'elo_winner','elo_loser', 'proba_elo']]
    mask = np.triu(np.ones_like(df_num.corr(), dtype=bool))
    sns.heatmap(df_num.corr(),vmin=-1, vmax=1, annot=True,cmap='BrBG', mask = mask)
    plt.title("Correlation Heatmap")
    st.pyplot()
        
elif page == pages[3]:
    
    df2 = df.copy()
    df2.reset_index(drop=True, inplace=True)
    df_nan = df2.loc[(df2["PSW"].isna())&(df2["B365W"].isna())]
    df2.drop(df_nan.index, inplace = True)
    
    df2["PSW"].fillna(df2["B365W"], inplace =True)
    df2["PSL"].fillna(df2["B365L"], inplace =True)
    df2["B365W"].fillna(df2["PSW"], inplace =True)
    df2["B365L"].fillna(df2["PSL"], inplace =True)
    del_var = ["ATP","Tournament","Comment","Date","Location"]
    df2.drop(del_var,axis=1, inplace = True)
    df2["Cote_W"] = (df2.PSW + df2.B365W)/2
    df2["Cote_L"] = (df2.PSL + df2.B365L)/2
    to_drop = ["PSW","PSL","B365W","B365L", "Wsets", "Lsets"]
    df2.drop(to_drop, axis=1, inplace = True)
    Surface = pd.crosstab(df2["Loser"],columns = df2["Surface"], values = df2["Surface"], aggfunc="count")
    Surface = Surface.fillna(0)
    Surface["Best_surface_Loser"] = Surface.idxmax(axis = 1)
    to_drop = ["Carpet","Clay","Grass","Hard"]
    Surface.drop(to_drop, inplace = True, axis = 1)

    Surface_2 = pd.crosstab(df2["Winner"], columns =df2["Surface"],values =df2["Surface"], aggfunc="count")
    Surface_2 = Surface_2.fillna(0)
    Surface_2["Best_surface_winner"] = Surface_2.idxmax(axis=1)
    to_drop = ["Carpet","Clay","Grass","Hard"]
    Surface_2.drop(to_drop, inplace = True, axis = 1)

    df2 = pd.merge(df2, Surface, how="left", on = "Loser")
    df2 = pd.merge(df2, Surface_2, how ="left", on ="Winner")
    
    import random

    df2["Player_1"] = df2["Winner"]
    df2["Player_2"] = df2["Loser"]
    joueurs = ["Player_1", "Player_2"]
    df2["Winner"] = df2["Winner"].apply(lambda x : random.choice(joueurs))
    df2["Loser"] = df2["Winner"].apply(lambda x: "Player_2" if x == "Player_1" else "Player_1")
    df2.to_csv("new_df2.csv")
    
    
    
    #Création de deux variables refletant le elo de player_1 et player_2
    elo_player_1 = []
    elo_player_2 = []

    for i in range(len(df2["Winner"])):
        if df2["Winner"][i] == "Player_1":
            elo_player_1.append(df2["elo_winner"][i])
            elo_player_2.append(df2["elo_loser"][i])
        else:
            elo_player_1.append(df2["elo_loser"][i])
            elo_player_2.append(df2["elo_winner"][i])
    #Création de deux variables refletant le rang de player_1 et player_2
    rank_player_1 = []
    rank_player_2 = []

    for i in range(len(df2["WRank"])):
        if df2["Winner"][i] == "Player_1":
            rank_player_1.append(df2["WRank"][i])
            rank_player_2.append(df2["LRank"][i])
        else:
            rank_player_1.append(df2["LRank"][i])
            rank_player_2.append(df2["WRank"][i])
    #Création de deux variables contenant les cotes du player_1 et du player_2
    Cote_player_1 =[]
    Cote_player_2 = []

    for i in range(len(df2)):
        if df2["Winner"][i] == "Player_1":
            Cote_player_1.append(df2["Cote_W"][i])
            Cote_player_2.append(df2["Cote_L"][i])
        else :
            Cote_player_1.append(df2["Cote_L"][i])
            Cote_player_2.append(df2["Cote_W"][i])

    # Création de deux variables donnant le type de surface favoris du player_1 et du player_2

    Surface_player_1 =[]
    Surface_player_2 = []

    for i in range(len(df2)):
        if df2["Winner"][i] == "Player_1":
            Surface_player_1.append(df2["Best_surface_winner"][i])
            Surface_player_2.append(df2["Best_surface_Loser"][i])
        else:
            Surface_player_1.append(df2["Best_surface_Loser"][i])
            Surface_player_2.append(df2["Best_surface_winner"][i])
    #Ajoutons ces nouvelles variables au dataset
    df2["elo_player_1"] = elo_player_1
    df2["elo_player_2"] = elo_player_2
    df2["rank_player_1"] = rank_player_1
    df2["rank_player_2"] = rank_player_2
    df2["Cote_player_1"] = Cote_player_1
    df2["Cote_player_2"] = Cote_player_2
    df2["Surface_player_1"] = Surface_player_1
    df2["Surface_player_2"]= Surface_player_2
    #Suppression des colonnes non utiles
    to_drop = ["WRank","LRank","elo_winner","elo_loser","Cote_W","Cote_L","Best_surface_winner","Best_surface_Loser"]
    df2.drop(to_drop, axis = 1, inplace=True)
    #Appliquons la formule pour créer la première variable prob_player_1
    prob_player_1 = []
    for i in range(len(df2["elo_player_1"])):
        d= df2["elo_player_1"][i] - df2["elo_player_2"][i]
        prob = 1/ (1 + 10**(-d / 400))
        prob_player_1.append(prob)

    #Ajoutons les deux nouvelles variables au dataset puis renommons la colonnes Winner en final_winner
    df2["prob_player_1"] = prob_player_1
    df2["Final_winner"] = df2["Winner"]

    #Supprimons les variables non utiles
    to_drop = ["Winner","Loser","proba_elo"]
    df2.drop(to_drop, axis = 1, inplace = True)

    st.write("# Machine Learning")
    st.write("Notre but ici est de prédire le vainqueur d'un match de tennis, nous avons donc un problème de Classification : Joueur 1 ou Joueur 2")
    st.write("##### 1. Suppression des variables non-utiles et Feature engineering")
    st.write("- Plusieurs colonnes ne seront pas utiles à notre projet. Nous prenons la décision de les supprimer. Ainsi les Variables **ATP, Tournament,Comment,Date,Location, Wsets et Lsets** ont été retirés de notre dataset.")
    st.write("- Ensuite, nous décidons de créer de deux nouvelles colonnes **Cotes_W et Cote_L** regroupant la moyenne de PSW/B365W - PSL/B365L, puis supprimons **PSW, PSL, B365W et B365L**. Cela nous permet d'avoir les cotes moyennes proposées par les bookmakers afin de réduire le nombre de variables sans perte d'informations.")
    st.write("- Nous décidons d'ajouter deux nouvelles variables **Surface_winner** et **Surface_looser** qui représentent le type de terrain favoris de chaque joueur")
    
    st.write("##### 2. Randomisation de notre dataset")
    st.write(""" ***Petite anecdote avant de continuer pour expliquer ce point de randomisation.
             Au début du lancement de l'étape de machine learning après avoir encodé et standardisé notre dataset nous avons commencé à entrainer notre dataset avec différents algorithmes
             tel que la regression logistique, le Knn, Svc, le decision Tree et le random forest. 
             Nos résultats étaient très performant ... trop performant avec des scores de 0.97 en moyenne. Ce sur-apprentissage est dut au fait que nos variables indiquent pour la plupart le vainqueur du match. Par exemple Elo_winner ou Cote_W. Nous avons donc compris 
             que le réel défis de ce projet était dans un premier temps définir notre variable cible qui devra prédire le gagnant d'un match puis de randomiser notre dataset afin qu'aucune variable n'indique qui a gagné un match de manière évidente afin de supprimer ce biais de sur-apprentissage.***""")
    
    st.write("""Notre dataset est nettoyé et vide de toute valeur nulle. Cependant, nous devons le transformer afin d'effectuer du machine learning dessus. En effet, nous souhaitons prédire les vainqueurs des matchs de tennis afin de parier et de battre les bookmakers. Pour cela plusieurs transformations de variables sont necessaires.
La colonne Winner est notre cible, c'est celle que nous souhaitons prédire. Nous avons remplacé ses valeurs par player_1 et player_2 aléatoirement. A partir de là, les colonnes WRank, LRank, elo_winner, elo_loser, Cote_W et Cote_L doivent être changées car elles nous donnent les vainqueurs du match or si nous utilisons un modèle de ML il utilisera ses données qui ne nous serons pas accessible avant l'issu du match et donc aura des résultats très performants sur les données d'entraînement et de test mais sera incapable de prédire sur des nouvelles données.
Nous devons donc créer de nouvelles variables "neutres" qui pourront être utilisées dans nos modèles de ML.""")
    
    st.write("""##### 3. Variable Proba_elo""")
    st.write("""Après une recherche sur internet nous pouvons comprendre que cette variable est un calcul de probabilité de victoire basé sur le elo de chaque joueur.
Le calcul est le suivant: \n
**P(A gagne contre B) = 1 / (1 + 10^(-d / 400))** \n
Où **d** est égale à la différence de elo entre les deux joueur.
Par exemple, si le joueur A a un classement Elo de 2000 points et le joueur B un classement de 1800 points, la différence (d) est de 200 points.
En appliquant la formule, on obtient :
P(A gagne contre B) = 1 / (1 + 10^(-200 / 400)) = 0.64
Maintenant que nous avons compris cela nous pouvons donc créer deux variables prob_player_1 et prob_player_2 indiquant la probabilité que chaque joueur a de gagner le match.""")
    
    st.write("##### 4. Dataset Final avant encodage et standardisation")
    
    st.write(df2)
    def get_data_summary(df2):
        summary_df = pd.DataFrame(index=df2.columns, columns=[ "Unique Values", "Missing Values", "Data Type", "% Missing"])

        summary_df["Unique Values"] = df2.nunique()
        summary_df["Missing Values"] = df2.isna().sum()
        summary_df["Data Type"] = df2.dtypes
        summary_df["% Missing"] = round((df2.isna().sum() / len(df2)) * 100, 2)

        return summary_df

    if st.checkbox("Show Missing Values Summary"):
        summary_df = get_data_summary(df2)
        st.dataframe(summary_df.astype(str))
        
    st.write("##### 5. Standardisation et encodage")
    df3 = df2.copy()
    df3_num = df3[["Best of","elo_player_1","elo_player_2","rank_player_1","rank_player_2","Cote_player_1","Cote_player_2","prob_player_1"]]
    df3_cat = df3[["Series",	"Court",	"Surface",	"Round","Surface_player_1",	"Surface_player_2"]]
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    df3_num_sc = sc.fit_transform(df3_num)
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    ohe = OneHotEncoder(sparse_output=False)
    df3_ohe = ohe.fit_transform(df3_cat[["Series",	"Court",	"Surface","Surface_player_1","Surface_player_2"]])
    df3_le = le.fit_transform(df3_cat["Round"])
    df3_le_df = pd.DataFrame(df3_le, columns=["Round"])
    df3_ohe_df = pd.DataFrame(df3_ohe, columns=ohe.get_feature_names_out())
    df3_num_df = pd.DataFrame(df3_num_sc,columns = sc.get_feature_names_out())
    df_final = pd.concat([df3_ohe_df,df3_le_df,df3_num_df, df2["Final_winner"]], axis = 1)
    
    st.write("""Dernière étape avant de procéder à la mise en place de nos algorithmes.\n Nous encodons nos variables catégoriels : **"Series",	"Court",	"Surface",	"Round","Surface_player_1",	"Surface_player_2"**. Puis nous standardisons nos variables numériques : **"Best of", "Year", "elo_player_1", "elo_player_2", "rank_player_1", "rank_player_2", "Cote_player_1", "Cote_player_2", "prob_player_1"**""")
    st.write(" Et voici notre dataset final une fois réunis: ")
    st.write(df_final)
    st.write(df_final.shape)

elif page == pages[4]:
    st.write("# Résultats de nos modèles")
    st.write("Selectionnez un modèle, le résultats sera celui obtenu avec les meilleurs hyperparamètres trouvés avec un GridsearchCv:")
    
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split
    
    df_final = pd.read_csv("df_final_ML.csv")
    feats = df_final.drop("Final_winner", axis=1)
    target = df_final["Final_winner"]
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.20,shuffle = False)
      
    model = st.selectbox(label = "Model", options= ["Logistic regression", "Knn", "Decision tree","Random forest"])
    from sklearn.metrics import r2_score
    
    def train_model(model):
        # sourcery skip: inline-immediately-returned-variable, switch
        if model == "Logistic regression" :
            return "0.70479", " Hyperparamètres : {'C': 0.1}"
        elif model == "Decision tree" : 
            return "0.70345", "Hyperparamètres : {'max_depth': 1, 'min_samples_leaf': 1, 'min_samples_split': 2}"
        elif model == "Knn":
            return "0.66952",  "Hyperparamètres : {'metric': 'manhattan', 'n_neighbors': 13, 'weights': 'uniform'}"
        elif model == "Random forest":
            return  " 0.70453", "Hyperparamètres : {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 100}"


        
    st.write("Score de prédiction :", train_model(model))
    st.write("Et voici les variables les plus importantes dans notre modèle de Regression logistic avec qui nous obtenons les meilleurs résultats")
    st.image("Importance_cote.png" )
   
            
elif page == pages[5] :
    st.write("# Simulation")
    st.write("Maintenant que nous avons trouvé notre modèle le plus performant il s'agit de savoir si nous pouvons parier de façon à gagner de l'argent")
    st.write("Partons de notre dataset et sélectionnons l'année 2018 par exemple.")
    st.write("Sélectionnez une cote minimum afin de filtrer les match sur lesquelles on ne pariera pas, puis sélectionnez un indice de confiance. Pour rappel plus l'indice de confiance est proche de **0** plus l'indice est faible et plus on pariera. A l'inverse plus la cote est proche de **1** moins de matchs seront jouables")
   
    
    
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Chargement des données
    df = pd.read_csv("df_ml.csv")

    # Division en ensembles d'entraînement et de test (2018 comme année de test)
    df_train = df[df['Year'] != 2018].copy()
    df_test = df[df['Year'] == 2018].copy()

    # Réinitialisation des index pour éviter les problèmes d'alignement ultérieurs
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    # Prétraitement des données (OneHotEncoder, LabelEncoder, StandardScaler)
    categorical_cols = ["Series", "Court", "Surface"]
    numeric_cols = ["Best of", "Year", "elo_player_1", "elo_player_2", "rank_player_1", "rank_player_2", "prob_player_1", "Cote_player_1", "Cote_player_2"]

    # OneHotEncoder pour les variables catégorielles
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe_train = ohe.fit_transform(df_train[categorical_cols]).toarray()
    ohe_test = ohe.transform(df_test[categorical_cols]).toarray()
    ohe_train_df = pd.DataFrame(ohe_train, columns=ohe.get_feature_names_out(categorical_cols))
    ohe_test_df = pd.DataFrame(ohe_test, columns=ohe.get_feature_names_out(categorical_cols))

    # LabelEncoder pour la variable "Round"
    le = LabelEncoder()
    df_train['Round_encoded'] = le.fit_transform(df_train['Round'])
    df_test['Round_encoded'] = le.transform(df_test['Round'])

    # StandardScaler pour les variables numériques
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(df_train[numeric_cols])
    scaled_test = scaler.transform(df_test[numeric_cols])
    scaled_train_df = pd.DataFrame(scaled_train, columns=numeric_cols)
    scaled_test_df = pd.DataFrame(scaled_test, columns=numeric_cols)

    # Combinaison des données prétraitées
    X_train = pd.concat([ohe_train_df, scaled_train_df, df_train['Round_encoded']], axis=1)
    X_test = pd.concat([ohe_test_df, scaled_test_df, df_test['Round_encoded']], axis=1)
    y_train = df_train['Final_winner']
    y_test = df_test['Final_winner']

    # Entraînement du modèle
    model = RandomForestClassifier(max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=100)
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] 
    
    
    cote = st.selectbox(label = "Choix de la cote minimum", options= [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.5])
    confidence = st.selectbox(label = "Indice de confiance", options = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    mise = st.selectbox(label = "Mise", options = [10,15,20,25,30,35,40,50,55,60,65,70,75,80,85,90,95,100])
    
    
    def simulation(y_pred, y_pred_proba, df_test, cote, confidence, mise):
        gain = 0
        match_paris = []
        match_results = []  # Initialize empty list to store match results

        for i, proba in enumerate(y_pred_proba):
            # Create a dictionary to store match data
            match_data = {
                'Player_1': df_test['Player_1'].iloc[i],
                'Player_2': df_test['Player_2'].iloc[i],
                'Round': df_test['Round'].iloc[i],
                'Series': df_test['Series'].iloc[i],
                'Prediction': y_pred[i],
                'Final_winner': df_test['Final_winner'].iloc[i],
                'Correct': y_pred[i] == df_test['Final_winner'].iloc[i]
            }

            if 0.5-(confidence/2) <= proba <= 0.5+(confidence/2) or (y_pred[i] == "Player 1" and df_test["Cote_player_1"][i]<=cote) or (y_pred[i] == "Player 2" and df_test["Cote_player_2"][i]<=cote):
                gain += 0
                match_paris.append(0)
            elif y_pred[i] == df_test["Final_winner"].iloc[i] and y_pred[i] == "Player_1":
                gain += mise * df_test["Cote_player_1"][i] 
                match_paris.append(1)
            elif y_pred[i] == df_test["Final_winner"].iloc[i] and y_pred[i] == "Player_2":
                gain += mise * df_test["Cote_player_2"][i] 
                match_paris.append(1)
            else:
                gain -= mise
                match_paris.append(1)

            match_results.append(match_data)  

        df_results = pd.DataFrame(match_results) 
        return gain, match_paris, df_results

    gain, match_paris, df_results = simulation(y_pred, y_pred_proba, df_test, cote, confidence, mise)

    st.write("Le score de précision de la simulation est de :", round(model.score(X_train, y_train)*100,3),"% de match correctement prédis.")
    st.write("\nGains estimés :",round(gain,2)," Euros sur", sum(match_paris),"matchs pariés au total")
    st.write("En 2018 au total",len(df_test), "matchs ont été joués")
       
    st.write("Résultats des matchs filtrés et prédictions:")
    st.write(df_results)

    
       

   

   

        
elif page == pages[6] :
    Mak = "Mak.jpg"
    st.image(Mak, width= 200 )
    st.write("### AL KUBAISI Mehdi") 
    st.write("""*Optométriste de formation, j'ai travaillé plus de huit and dans l'optique en France et au Canada. 
             J'ai ensuite ouvert un coffee shop sous franchise dans ma ville natale du Havre pendant sept ans. Passioné par tout ce qui touche au numérique j'ai décidé de suivre une formation dans
             le domaine de la data avec Datascientest afin de devenir Analiste de données. Cette reconversion est une nouvelle étape dans mon parcours un peu atypique qui je l'espère sera
             porteuse de nouvelles opportunités et de nouveaux défis à relever !*""")
    
    Jonhatan = "jonathan.jpg"
    st.image(Jonhatan, width= 250 )
    st.write("### CHICHEPORTICHE jonathan") 
    st.write("""*Doté de 7 ans d'expérience dans l'événementiel, spécialisé dans les grands salons français, j'ai réorienté ma carrière vers ma formation initiale en comptabilité en 2016. Soucieux d'évoluer et de m'adapter aux nouvelles exigences du marché, j'ai décidé de me former en tant que data analyst. Cette formation me permet d'acquérir des compétences en exploration de données afin de les appliquer dans mon travail et d'identifier de nouvelles opportunités.*""")
    
    Amar = "Amar.jpg"
    st.image(Amar, width= 200 )
    st.write("### ACHOUR Amar") 
    st.write("""*Historien de formation, je me suis redirigé vers l’intelligence économique en effectuant un master en veille et technologies de l’information. Le monde professionnel m’a permis de me former au scraping. Aujourd’hui, avec les nouveaux enjeux de la donnée, c’est naturellement que j’ai choisi la formation de Data Analyst chez Datascientest pour être opérationnel sur tous les aspects de la data : collecte, visualisation et interprétation.*""")