import sqlite3


def setup_database():
    conn = sqlite3.connect("research_papers.db")
    cur = conn.cursor()

    # Create papers table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT,
            title TEXT NOT NULL,
            year INTEGER,
            url TEXT,
            read BOOLEAN DEFAULT FALSE
        )
    """)

    # Create paper_references table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS paper_references (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id INTEGER NOT NULL,
            referenced_paper_id TEXT NOT NULL,
            FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
            FOREIGN KEY (referenced_paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    cur.close()
    conn.close()


def insert_paper(paper_id, title, year=None, url=None, read=False):
    conn = sqlite3.connect("research_papers.db")
    cur = conn.cursor()

    # if not exists
    cur.execute(
        """
        SELECT id FROM papers WHERE paper_id = ?
    """,
        (paper_id,),
    )
    result = cur.fetchone()

    if result is None:
        cur.execute(
            """
            INSERT INTO papers (paper_id, title, year, url, read)
            VALUES (?, ?, ?, ?, ?)
        """,
            (paper_id, title, year, url, read),
        )

    paper_db_id = cur.lastrowid

    conn.commit()
    cur.close()
    conn.close()

    return paper_db_id


def insert_reference(paper_db_id, referenced_paper_id):
    conn = sqlite3.connect("research_papers.db")
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO paper_references (paper_id, referenced_paper_id)
        VALUES (?, ?)
    """,
        (paper_db_id, referenced_paper_id),
    )

    conn.commit()
    cur.close()
    conn.close()


def add_reference_paper_if_not_exists(
    referenced_paper_id, referenced_title, referenced_year=None, referenced_url=None
):
    conn = sqlite3.connect("research_papers.db")
    cur = conn.cursor()

    # Check if the referenced paper already exists
    cur.execute(
        """
        SELECT id FROM papers WHERE paper_id = ?
    """,
        (referenced_paper_id,),
    )
    result = cur.fetchone()

    if result is None:
        # Insert the referenced paper if it does not exist
        cur.execute(
            """
            INSERT INTO papers (paper_id, title, year, url)
            VALUES (?, ?, ?, ?)
        """,
            (referenced_paper_id, referenced_title, referenced_year, referenced_url),
        )

        paper_db_id = cur.lastrowid
    else:
        paper_db_id = result[0]

    conn.commit()
    cur.close()
    conn.close()

    return paper_db_id


def fetch_papers():
    conn = sqlite3.connect("research_papers.db")
    cur = conn.cursor()

    cur.execute("SELECT * FROM papers")
    papers = cur.fetchall()

    cur.close()
    conn.close()

    return papers


def fetch_references():
    conn = sqlite3.connect("research_papers.db")
    cur = conn.cursor()

    cur.execute("SELECT * FROM paper_references")
    references = cur.fetchall()

    cur.close()
    conn.close()

    return references


def get_most_referenced_paper_details():
    conn = sqlite3.connect("research_papers.db")
    cur = conn.cursor()

    # SQL query to find the most referenced paper by the read papers
    cur.execute("""
        SELECT p_references.referenced_paper_id, COUNT(*) AS reference_count
        FROM papers AS p_read
        JOIN paper_references AS p_references
        ON p_read.id = p_references.paper_id
        WHERE p_read.read = TRUE
        GROUP BY p_references.referenced_paper_id
        ORDER BY reference_count DESC
        LIMIT 1
    """)

    result = cur.fetchone()

    if not result:
        print("No references found from read papers.")
        cur.close()
        conn.close()
        return

    referenced_paper_id = result[0]
    reference_count = result[1]

    # Get the details of the most referenced paper
    cur.execute(
        """
        SELECT title, year, url
        FROM papers
        WHERE id = ?
    """,
        (referenced_paper_id,),
    )

    referenced_paper = cur.fetchone()

    if not referenced_paper:
        print(f"Referenced paper ID {referenced_paper_id} not found in papers table.")
        cur.close()
        conn.close()
        return

    referenced_paper_title, referenced_paper_year, referenced_paper_url = (
        referenced_paper
    )

    print(
        f"The most referenced paper by your read papers is:\n"
        f"Title: {referenced_paper_title}\n"
        f"Year: {referenced_paper_year}\n"
        f"URL: {referenced_paper_url}\n"
        f"Referenced {reference_count} times by the following papers:\n"
    )

    # Get the list of read papers that reference the most referenced paper
    cur.execute(
        """
        SELECT p_read.title, p_read.year, p_read.url, p_read.read
        FROM papers AS p_read
        JOIN paper_references AS p_references
        ON p_read.id = p_references.paper_id
        WHERE p_references.referenced_paper_id = ?
    """,
        (referenced_paper_id,),
    )

    referencing_papers = cur.fetchall()

    print("Read Papers:")
    for referencing_paper in referencing_papers:
        (
            referencing_paper_title,
            referencing_paper_year,
            referencing_paper_url,
            is_read,
        ) = referencing_paper
        if is_read:
            print(
                f"  - Title: {referencing_paper_title}\n"
                f"    Year: {referencing_paper_year}\n"
                f"    URL: {referencing_paper_url}"
            )

    cur.close()
    conn.close()
