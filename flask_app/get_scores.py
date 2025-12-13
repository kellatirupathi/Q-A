import pandas as pd

def get_final_scores(csv_file):
    try:
        df = pd.read_csv(csv_file)
        technical_df = df[df['tech_non_tech'] == 'Technical']
        score = technical_df[['question_text', 'answer_relevancy_score']].to_dict(orient='records')
        top_concepts = technical_df['question_concept'].value_counts().nlargest(4).index.tolist()
        technical_df.loc[~technical_df['question_concept'].isin(top_concepts), 'question_concept'] = 'Other'
        concept_avg_score = technical_df.groupby('question_concept')['answer_relevancy_score'].mean().to_dict()
        concept_avg_score_list = sorted([{"concept_name": concept, "average_score": round(score,2)} if concept != 'Other'
                                        else {"concept_name": concept, "average_score": score}
                                        for concept, score in concept_avg_score.items()],
                                        key=lambda x: x['concept_name'] == 'Other')
        return score, concept_avg_score_list
    except:
       print("Unable to get consolidated scores")
       return [],[]


  


