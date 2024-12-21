import os
import random
import re
import sys
import copy
import math

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    # corpus --> {'1.html': {'2.html'}, '2.html': {'3.html', '1.html'}, '3.html': {'4.html', '2.html'}, '4.html': {'2.html'}}
    # page_links --> {'3.html', '1.html'} for 2.html

    p_dist = dict()
    p_dist[page] = 0.0
    page_links = corpus[page]

    # If page has no outgoing links, then transition_model should return
    # a probability distribution that chooses randomly among all pages with equal probability.

    if len(page_links) == 0:
        for page in corpus:
            p_dist[page] = 1 / len(corpus)

    else:
        for p in corpus:
            # divide 1-d among all pages
            p_dist[p] = (1-damping_factor) / len(corpus)
            # div d among only the page links
            if p in page_links:
                p_dist[p] += damping_factor / len(corpus[page])

    return p_dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    page_p_dist = dict()
    page_rank = dict()
    num_visits = dict()
    pages_list = list(corpus)

    for page in corpus:
        num_visits[page] = 0

    r_page = random.choice(list(corpus))
    page_p_dist = transition_model(corpus, r_page, damping_factor)
    num_visits[r_page] += 1

    # initial population and weights to start with
    pages_list = list(page_p_dist)
    weights = list(page_p_dist.values())

    for i in range(SAMPLES-1):
        # choose a random page based on probability distribution (weight) from previous sample
        r_page = random.choices(pages_list, weights=weights, k=1)[0]

        page_p_dist = transition_model(corpus, r_page, damping_factor)
        num_visits[r_page] += 1

        # population and weights (prob distribution) for next sampling
        pages_list = list(page_p_dist)
        weights = list(page_p_dist.values())

    for page in num_visits:
        page_rank[page] = num_visits[page] / SAMPLES

    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_rank = dict()
    new_page_rank = dict()

    for page in corpus:
        page_rank[page] = 1 / len(corpus)

    repeat = True

    while repeat:
        for page in corpus:
            p_r = (1 - damping_factor) / len(corpus)
            summation = 0

            for link_page, links in corpus.items():
                # Page with no links interpreted as having one link for every page in corpus
                if not links:
                    links = corpus.keys()
                if page in links:
                    summation += page_rank[link_page] / len(links)

            p_r = p_r + damping_factor * summation
            new_page_rank[page] = p_r

        repeat = False

        # If any of the values changes by more than the threshold (0.001), repeat rank calculation
        for p in page_rank:
            if not math.isclose(new_page_rank[p], page_rank[p], abs_tol=0.001):
                repeat = True
                page_rank[p] = new_page_rank[p]

    return page_rank


if __name__ == "__main__":
    main()
