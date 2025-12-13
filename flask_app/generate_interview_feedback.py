from jinja2 import Environment, FileSystemLoader


def get_generated_intervew_html_path(candidate_name,first_letter,company_name,video_link,interview_date,duration,candidate_email,concept_scores,question_scores,feedback_timeline):

    candidate_image = "https://test-diarizaton.s3.ap-south-1.amazonaws.com/candidate_profile_image.webp"
    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template("interview_feedback.html")

    html_content = template.render(
        candidate_name=candidate_name,
        company_name=company_name,
        video_link=video_link,
        interview_date=interview_date,
        duration=duration,
        candidate_email=candidate_email,
        candidate_image=candidate_image,
        concept_scores=concept_scores,
        question_scores=question_scores,
        feedback_timeline=feedback_timeline,
        first_letter=first_letter
    )
    with open("candidate_interview_feedback.html", "w", encoding="utf-8") as html_file:
        html_file.write(html_content)
        return "candidate_interview_feedback.html"
