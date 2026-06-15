# AG News Classification

## Vue d'ensemble
AG News Classification est un jeu de données de classification de texte où l'objectif est de prédire la catégorie d'un article de presse à partir de son titre et de sa description.
La version couramment utilisée sur Kaggle contient quatre classes, **World**, **Sports**, **Business** et **Sci/Tech**.

Il contient 120 000 exemples d'entraînement et 7 600 exemples de test, avec une répartition équilibrée entre les classes.

## Le problème métier
Une plateforme média, un moteur de recherche, une application mobile ou un agrégateur d'actualités doit être capable de ranger automatiquement de grands volumes d'articles dans les bonnes rubriques.
Sans cette automatisation, la catégorisation doit être faite manuellement, ce qui devient vite coûteux, lent et incohérent dès que le flux d'articles augmente.

Concrètement, un système de classification de news sert à plusieurs choses :
- Alimenter les rubriques d'un site ou d'une app en temps réel, par exemple envoyer un article dans la section Business plutôt que Sports.
- Améliorer la recherche et la navigation, car un contenu bien catégorisé est plus facile à filtrer, recommander et retrouver.
- Personnaliser l'expérience utilisateur, par exemple pousser davantage d'articles Sci/Tech à un lecteur intéressé par la technologie.
- Aider à la modération éditoriale et à l'analytics, en mesurant la part de chaque thématique dans le flux publié.

En langage simple, le besoin métier est le suivant : quand un nouvel article arrive, le système doit comprendre rapidement de quoi il parle et le ranger au bon endroit. Techniquement, c'est un problème de classification multiclasse, car une entrée textuelle doit être assignée à une classe parmi plusieurs catégories possibles.

## Variables et cible
Chaque observation contient en général trois colonnes principales : un label numérique, un titre et une description de l'article. La cible représente la catégorie à prédire, tandis que les colonnes textuelles servent d'entrée au modèle.

| Élément | Rôle |
|---|---|
| `Class Index` | Label de classification à prédire. |
| `Title` | Titre de l'article, souvent très informatif pour la classe. |
| `Description` | Résumé ou courte description, utile pour lever les ambiguïtés. |

