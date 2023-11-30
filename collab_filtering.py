import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class UserCF:
    def __init__(self, id):
        """"initialization my userId, number of sumilarity users and similarity threshold"""
        self.userId = id
        self.number_of_sim_users = 10
        self.user_similarity_threshold = 0.3

    def get_data(self):
        """get a movie rating and leave only those with more than 100 ratings"""
        df = pd.read_csv("dataset/for_userCF.csv")

        agg_df = (df.groupby("title").agg(
            number_of_rating=("rating", "count"), mean_of_rating=("rating", "mean")
            ).reset_index())

        agg_df_gt100 = agg_df[agg_df["number_of_rating"] >= 100]

        # for test i use userid greater than 1000
        self.df_gt100 = pd.merge(
            df[df["userId"] < 1000], agg_df_gt100["title"], on="title", how="inner"
        )

    def calc_user_sim(self):
        """Calculation of similar users using Pearson correlation"""
        matrix = self.df_gt100.pivot_table(
            index="userId", columns="title", values="rating"
        )
        # Normalization
        self.matrix_norm = matrix.subtract(matrix.mean(axis=1), axis="rows")

        user_similarity = pd.DataFrame(
            cosine_similarity(self.matrix_norm.fillna(0)),
            index=self.matrix_norm.index,
            columns=self.matrix_norm.index,
        )
        # delete the line with my user id
        user_similarity.drop(index=self.userId, inplace=True)

        self.similar_users = user_similarity[user_similarity[self.userId] > self.user_similarity_threshold][self.userId].sort_values(ascending=False)[:self.number_of_sim_users]

    def narrow_down_item_pool(self):
        """remove movies I've already watched"""
        picked_userid_watched = self.matrix_norm[
            self.matrix_norm.index == self.userId
        ].dropna(axis=1, how="all")

        self.similar_user_movies = self.matrix_norm[
            self.matrix_norm.index.isin(self.similar_users.index)
        ].dropna(axis=1, how="all")

        self.similar_user_movies.drop(
            picked_userid_watched.columns, axis=1, inplace=True, errors="ignore"
        )

    def get_rec(self):
        item_score = {}
        # Loop through items
        for i in self.similar_user_movies.columns:
            # Get the ratings for movie i
            movie_rating = self.similar_user_movies[i]
            # Create a variable to store the score
            total = 0
            # Create a variable to store the number of scores
            count = 0
            # Loop through similar users
            for u in self.similar_users.index:
                # If the movie has rating
                if not pd.isna(movie_rating[u]):
                    # Score is the sum of user similarity score multiply by the movie rating
                    score = self.similar_users[u] * movie_rating[u]
                    # Add the score to the total score for the movie so far
                    total += score
                    # Add 1 to the count
                    count += 1
            # Get the average score for the item
            item_score[i] = total / count
        # Convert dictionary to pandas dataframe
        item_score = pd.DataFrame(item_score.items(), columns=["movie", "movie_score"])

        # Sort the movies by score
        ranked_item_score = item_score.sort_values(by="movie_score", ascending=False)
        # Select top m movies
        m = 10
        print(ranked_item_score(m))


if __name__ == "__main__":
    userCF = UserCF(1)

    userCF.get_data()
    userCF.calc_user_sim()
    userCF.narrow_down_item_pool()
    userCF.get_rec()
