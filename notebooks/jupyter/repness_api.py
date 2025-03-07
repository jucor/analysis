#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import os
import dotenv
import json

dotenv.load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
ZID = os.getenv("ZID")
print("DB_URL", DB_URL)

def fix_postgres_url_for_sqlalchemy(db_url):
    if db_url and db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    return db_url

def get_math_blob(db_url, zid):
    db_url = fix_postgres_url_for_sqlalchemy(db_url)

    """Get math data for a conversation."""
    query = f"""
    SELECT data as json_blob, math_tick, last_vote_timestamp, modified 
    FROM math_main 
    WHERE zid = {zid}
    """
    df = pd.read_sql(query, db_url)
    if len(df) == 0:
        raise ValueError(f"No math data found for zid: {zid}")
    return df.iloc[0]["json_blob"]


def get_groups(db_url, zid):
    db_url = fix_postgres_url_for_sqlalchemy(db_url)
    # Get math data to extract clusters
    math_data = get_math_blob(db_url, zid)

    # Get clusters from math data
    group_clusters = math_data.get("group-clusters", [])
    base_clusters = math_data.get("base-clusters", {})

    # Create a mapping from cluster ID to index in the array
    cluster_id_to_index = {cluster_id: idx for idx, cluster_id in enumerate(base_clusters["id"])}

    # Collect all participants for each group based on their associated base clusters
    group_ptpts = []
    for group in group_clusters:
        group_ptpt_set = set()
        for cluster_id in group["members"]:
            # Find the index for this cluster ID in the parallel arrays
            if cluster_id in cluster_id_to_index:
                cluster_index = cluster_id_to_index[cluster_id]
                # Add the participants from this cluster to the group
                if cluster_index < len(base_clusters["members"]):
                    group_ptpt_set.update(base_clusters["members"][cluster_index])
                else:
                    raise ValueError(
                        f"Cluster index {cluster_index} out of bounds for members array of length {len(base_clusters['members'])}"
                    )
            else:
                raise ValueError(f'Cluster ID {cluster_id} not found in base_clusters["id"]')

        group_ptpts.append(list(group_ptpt_set))
    return group_ptpts


def get_data_from_db(db_url, zid):

    db_url = fix_postgres_url_for_sqlalchemy(db_url)

    votes_query = f"""SELECT 
    pid as "participant", 
    tid as "comment-id", 
    created as "timestamp",
    -1*vote as "vote"
    FROM votes WHERE zid = {zid} ORDER BY pid, tid"""
    votes_df = pd.read_sql(votes_query, con=db_url)

    # Check for duplicate participant-comment pairs and only keep the latest vote
    # First sort the dataframe by timestamp, then group and keep the last entry in each group
    votes_df = votes_df.sort_values(by="timestamp", ascending=True)
    votes_df = votes_df.drop_duplicates(subset=["participant", "comment-id"], keep="last")

    """Get comments from the database"""
    query = f"""SELECT 
        tid AS "comment-id",
        created AS "timestamp", 
        pid AS "author-id",
        mod as "moderated"
    FROM comments WHERE zid = {zid} ORDER BY "comment-id" DESC"""
    comments_df = pd.read_sql(query, con=db_url)

    # Group votes by comment-id and vote value, then pivot to get counts for each vote type
    vote_counts = votes_df.groupby(["comment-id", "vote"]).size().reset_index(name="count")

    # Pivot the table to have separate columns for each vote value
    vote_counts_pivot = vote_counts.pivot(
        index="comment-id", columns="vote", values="count"
    ).reset_index()

    # Rename columns for clarity
    vote_counts_pivot.columns = ["comment-id", "disagrees", "pass", "agrees"]
    # Drop the 'pass' column
    vote_counts_pivot = vote_counts_pivot.drop(columns=["pass"])

    # Merge comments with vote counts
    comments_df = comments_df.merge(vote_counts_pivot, on="comment-id", how="left")

    # Pivot the votes dataframe to have participants as rows and comments as columns
    votes_pivot_df = votes_df.pivot(index="participant", columns="comment-id", values="vote")

    # Convert vote values to integers
    for col in votes_pivot_df.columns[1:]:
        votes_pivot_df[col] = votes_pivot_df[col].astype(float)

    # Create a dataframe to store participant statistics
    participant_stats = pd.DataFrame(index=votes_pivot_df.index)

    # Count the number of votes for each participant (excluding NaN values)
    participant_stats["n-votes"] = votes_pivot_df.count(axis=1)

    # Count the number of agrees (vote=1) for each participant
    participant_stats["n-agree"] = (votes_pivot_df == 1).sum(axis=1)

    # Count the number of disagrees (vote=-1) for each participant
    participant_stats["n-disagree"] = (votes_pivot_df == -1).sum(axis=1)

    # Count the number of comments authored by each participant
    # First, create a Series counting comments per author
    comment_counts = comments_df.groupby("author-id").size()

    # Add the comment counts to participant_stats, filling NaN values with 0
    # (participants who didn't author any comments)
    participant_stats["n-comments"] = (
        participant_stats.index.map(lambda x: comment_counts.get(x, 0)).fillna(0).astype(int)
    )

    # Add a placeholder for group-id (will be filled later)
    participant_stats["group-id"] = np.nan

    # Reorder columns to have group-id first, followed by statistics
    participant_stats = participant_stats[
        ["group-id", "n-comments", "n-votes", "n-agree", "n-disagree"]
    ]
    # Add the participant stats to the votes_pivot_df
    votes_pivot_df = pd.concat([participant_stats, votes_pivot_df], axis=1)

    groups = get_groups(os.getenv("DATABASE_URL"), 17909)
    # Update group-id column based on the groups list
    # Initialize all group-ids as NaN
    votes_pivot_df["group-id"] = np.nan

    # For each group index and its participants
    for group_idx, participants in enumerate(groups):
        # Set the group-id for all participants in this group
        votes_pivot_df.loc[votes_pivot_df.index.isin(participants), "group-id"] = float(group_idx)

    return votes_pivot_df, comments_df



def count_finite(row):
    """for a row, count the number of finite values """
    finite = np.isfinite(row[val_fields])  # boolean array of whether each entry is finite
    return sum(finite)  # count number of True values in `finite`


def select_rows(df, threshold=7):
    """ REMOVE PARTICIPANTS WITH LESS THAN N VOTES check for each row if the number of finite values >= cutoff """

    number_of_votes = df.apply(count_finite, axis=1)
    valid = number_of_votes >= threshold

    return df[valid]


def main():
    df, df_comments = get_data_from_db(DB_URL, ZID)
    df_comments.head()
    print("Type of comment-id column:", df_comments["comment-id"].dtype)


    df_comments.index = df_comments.index.astype(str)



    metadata_fields = ["group-id", "n-comments", "n-votes", "n-agree", "n-disagree"]
    val_fields = [c for c in df.columns.values if c not in metadata_fields]

    # remove statements (columns) which were moderated out
    statements_all_in = sorted(list(df_comments.loc[df_comments["moderated"] > 0].index.array), key=int)


    df = select_rows(df)


    vals = df[val_fields]
    # If the participant didn't see the statement, it's a null value, here we fill in the nulls with zeros
    vals = vals.fillna(0)  # <---in paper: column mean
    vals = vals.sort_values("participant")
    vals.columns = vals.columns.astype(str)
    vals_all_in = vals[statements_all_in]


    # # Comment Statistics
    # 
    # We analyze comments for how strongly they represent each opinion group.
    # For that, the representative metric R_v(g,c) is calculated for all groups g, comments c, and possible votes v.
    # This metric estimates how much more likely participants in group g are vote v on said comment c than those outside group g.
    # 
    # _Definition of R_v(g,c):_
    # Let N*v(g,c) be the number of participants in group g who cast vote v on comment c, and let N(g,c) be the total number of votes for comment c within group g (i.e. <font color='red'> N*{+1}(g,c)+ N\_{-1}(g,c) <font> )
    # 
    # Defifne P_v(g,c)=(1+N_v(g,v))/(2+N(g,c)) which estimates the probability that a given person in group g votes v on comment c. Then
    # 
    # R_v(g,c) = P_v(g,c)/ P_v(g_not,c)
    # 
    # where g_not denotes the complement of g, thus all participants not in g.
    # 


    # Step 1: Calculate N_v(g,c), N(g,c), and P_v(g,c)
    N_groups = df["group-id"].nunique()
    N_comments = len(statements_all_in)
    N_v_g_c = np.zeros([3, N_groups, N_comments])  # create N matrix
    P_v_g_c = np.zeros([3, N_groups, N_comments])
    N_g_c = np.zeros([N_groups, N_comments])
    v_values = [-1, 0, 1]

    for g in range(N_groups):
        # get indices of cluster g; caution_ idx != participant id
        idx_g = np.where(df["group-id"] == g)[0]
        for c in range(N_comments):
            comment = statements_all_in[c]  # comment id
            df_c = vals_all_in[str(comment)].iloc[
                idx_g
            ]  # data frame: [participants of group g,comment c],
            for v in range(3):
                v_value = v_values[v]
                N_v_g_c[v, g, c] = (
                    df_c == v_value
                ).sum()  # counts all v_value votes in data frame df_c
            N_g_c[g, c] = (
                N_v_g_c[0, g, c] + N_v_g_c[2, g, c]
            )  # total votes corresponds to votes with +1 or -1

            for v in range(3):
                P_v_g_c[v, g, c] = (1 + N_v_g_c[v, g, c]) / (2 + N_g_c[g, c])

    # Step2: calculate R_v(g,c)
    R_v_g_c = np.zeros([3, N_groups, N_comments])
    for g in range(N_groups):
        for c in range(N_comments):
            for v in range(3):
                R_v_g_c[v, g, c] = (
                    P_v_g_c[v, g, c] / np.delete(P_v_g_c[v, :, c], g, 0).sum()
                )  # np.delete neglects all entries with group g


    # # Comment Selection criterion
    # 
    # Given R_v(g,c), how to decide weather comment c is representative for group g?
    # 
    # Remember: R_v(g,c)=2 means that comment c is 2 times more likely to be voted v in group g compared to all the other groups.
    # However, this does not tell us how significant this difference is (a very small likelihood multiplied by 2 is still a very small likelihood).
    # 
    # <font color='red'> 
    # As a measure of significance, we calculate the Fisher exact test. This quantity can be regarded as a measure of correlation between two random variables. In fact, it tests how significantly the obtained sample (in this case votes v of comment c in group g) is different from the 0 hypothesis (in this case, that votes v are drawn from a hypergeometric distribution with parameters given by including all groups on comment c). 
    # <font>
    # 
    # We weight this significance measure with R_v(g,c).
    # 

    import scipy.stats as stats
    from scipy.stats import hypergeom

    v_values = [-1, 0, 1]
    p_values = np.zeros([N_groups, N_comments, 3])
    for g in range(N_groups):
        idx_g = np.where(df["group-id"] == g)[0]
        idx_g_not = np.where(df["group-id"] != g)[0]
        for c in range(N_comments):
            comment = statements_all_in[c]  # comment id

            for v in range(3):
                v_value = v_values[v]
                N_v = (
                    vals_all_in[str(comment)] == v_value
                ).sum()  # totol number of v votes in comment c
                N_rest = (
                    vals_all_in[str(comment)]
                ).count() - N_v  # total number of votes =  number of participants

                df_c = vals_all_in[str(comment)].iloc[idx_g]  # get data frame of group g for comment c
                N_v_in_g = (df_c == v_value).sum()
                N_g = (df_c).count()

                [M, n, N] = [
                    N_rest + N_v,
                    N_v,
                    N_g,
                ]  # hypergeometric distribution parameters https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html
                x = range(N_v_in_g - 1, N_g + 1)
                prb = hypergeom.pmf(x, M, n, N).sum()  # calculates P(X>=N_v_in_g), i.e. p-value.
                p_values[g, c, v] = prb * R_v_g_c[v, g, c]


    representativeness_data = {
        "vote_idx": v_values,
        "groups_idx": list(range(0, N_groups)),
        "statements_idx": [int(i) for i in statements_all_in],
        # Nested lists in order: groups, statements, vote_values
        "repness-p-values": p_values.tolist()
    }

    print(json.dumps(representativeness_data, indent=4))