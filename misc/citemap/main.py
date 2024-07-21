import requests
from db import *


def request_paper(title):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    params = {
        "query": title,
        "limit": 20,
        "fields": "title,year,url,authors,references,references.title,references.year,references.url",
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return response


def save_paper(paper_id, title, year=None, url=None, read=False):
    paper_db_id = insert_paper(paper_id, title, year, url, read)
    return paper_db_id


def save_references(paper_db_id: int, references: list):
    for reference in references:
        referenced_paper_id = reference.get("paperId")
        referenced_title = reference.get("title")
        referenced_year = reference.get("year")
        referenced_url = reference.get("url")
        if referenced_paper_id:
            referenced_paper_db_id = add_reference_paper_if_not_exists(
                referenced_paper_id, referenced_title, referenced_year, referenced_url
            )
            insert_reference(paper_db_id, referenced_paper_db_id)


setup_database()

read_papers = [
    "Generative Adversarial Nets",
    "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks",
    "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks",
    "Denoising Diffusion Probabilistic Models",
    "Very Deep Convolutional Networks for Large-Scale Image Recognition",
    "Deep Residual Learning for Image Recognition",
    "Improved Training of Wasserstein GANs",
    "Denoising Diffusion Implicit Models",
    "Attention Is All You Need",
    "Language Models are Unsupervised Multitask Learners",
    "LLaMA: Open and Efficient Foundation Language Models",
]

# Example usage
for title in read_papers:
    response = request_paper(title)
    if response:
        # print(response)
        paper = response.get("data")[0]

        paper_id = paper.get("paperId")
        title = paper.get("title")
        year = paper.get("year")
        url = paper.get("url")
        references = paper.get("references")

        paper_db_id = save_paper(paper_id, title, year, url, read=True)
        save_references(paper_db_id, references)
    else:
        print(f"Paper not found: {title}")

get_most_referenced_paper_details()


# # Fetch and print papers
# papers = fetch_papers()
# print("Papers:")
# for paper in papers:
#     print(paper)

# # Fetch and print references
# references = fetch_references()
# print("\nReferences:")
# for reference in references:
#     print(reference)
