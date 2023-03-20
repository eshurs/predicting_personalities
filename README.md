# Predicting Personalities with Natural Language Processing

Overview: the goal of this project is to model an individual's personality traits using NLP algorithms on their written responses.

### Personality Traits Framework: Five-Factor *OCEAN* Model of Personality

The basis of personality traits used for this project are derived from the Five-Factor Model (Goldberg, 1990; McCrae & John, 1992; McCrae & Costa, 1987).
[Linked Paper](https://openpress.usask.ca/introductiontopsychology/chapter/personality-traits/#:~:text=The%20most%20widely%20used%20system,Extraversion%2C%20Agreeableness%2C%20and%20Neuroticism)

The five personality traits in question are described below:

![image](https://user-images.githubusercontent.com/28024140/222189312-4aa9e6bb-72fa-4330-ba88-b026df7a56f4.png)

*This project uses "Emotional Stability" as opposed to "Neuroticism" so that all traits can be visualized with the same a positive/negative attribute convention:* 
- Openness / Close-mindedness
- Conscientious / Unconscientious
- Extroverte / Introverted
- Agreeable / Disagreeable
- Emotionally Stable / Neurotic

### Methodology

This project analyzed written responses to create personality trait scores using the [spaCy](https://spacy.io/) API for natural language processing in order to compare sentences in written responses to essays to survey questions that assess an individual's scores in each area of the five factor framework. The [questionaire](https://www.kaggle.com/datasets/tunguz/big-five-personality-test) uses phrases and words associated with each trait, as well as their opposites (e.g. there are questions that evaluate both extrovertedness and introvertedness in the extroverted category). 
<br />
<br />
The spaCy library contains a sentence/phrase similarity score function that compares each word within a given sentence and returns a score from 0-1 based on how similar the strings of words are. The scores for the written responses were calculated and normalized by scoring each sentence in the paragraph through comparing it to each sentence in the questionaire for a trait, then divided by the number of questions for that particular trait. These scores were then printed and visualized in a bar graph.

### Project Application

One of the proposed applications of this project is to breakdown the personality traits of users on a dating app so that they could be matched with other users with more compatability (see [OK Cupid Dataset](https://www.kaggle.com/datasets/andrewmvd/okcupid-profiles)). 
<br />
<br />
Below is a possible user-facing visualization that would allow for a more transparent assessment of a proposed match so that all users can be more informed.
![Figure_1](https://user-images.githubusercontent.com/28024140/226335205-0ddd2654-cc98-4951-929d-29be9ca2fb63.png)
*see user 100 responses here: *[user_responses.txt](https://github.com/eshurs/predicting_personalities/files/11017856/user_responses.txt)
