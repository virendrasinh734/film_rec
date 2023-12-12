import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec


class UserCF:
    def __init__(self, id):
        """"initialization my userId, number of sumilarity users and similarity threshold"""
        self.userId = id
        self.number_of_sim_users = 10
        self.user_similarity_threshold = 0.3

    def get_data(self):
        """get a movie rating and leave only those with more than 100 ratings"""
        df_userCF = pd.read_csv("dataset/for_userCF.csv")[:10000]
        self.df_contentF = pd.read_csv("dataset/movies_metadata.csv", usecols=["title", "overview"])[:1000]

        agg_df_userCF = (df_userCF.groupby("title").agg(
            number_of_rating=("rating", "count"), mean_of_rating=("rating", "mean")
            ).reset_index())

        agg_df_userCF_gt100 = agg_df_userCF[agg_df_userCF["number_of_rating"] >= 100]

        # for test i use userid greater than 1000
        self.df_userCF_gt100 = pd.merge(
            df_userCF[df_userCF["userId"] < 1000], agg_df_userCF_gt100["title"], on="title", how="inner"
        )
        self.model = Doc2Vec.load("models/d2v.model")

    def calc_sim(self):
        """Calculation of similar users using Pearson correlation"""
        matrix = self.df_userCF_gt100.pivot_table(
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

        document_vectors = [self.model.infer_vector(word_tokenize(str(doc).lower())) for doc in self.df_contentF["overview"]]
        self.movie_similarity = pd.DataFrame(cosine_similarity(pd.DataFrame(document_vectors)), index=self.df_contentF["title"], columns=self.df_contentF["title"])

    def narrow_down_item_pool(self):
        """remove movies I've already watched"""
        self.picked_userid_watched = self.matrix_norm[
            self.matrix_norm.index == self.userId
        ].dropna(axis=1, how="all")

        self.similar_user_movies = self.matrix_norm[
            self.matrix_norm.index.isin(self.similar_users.index)
        ].dropna(axis=1, how="all")

        self.similar_user_movies.drop(
            self.picked_userid_watched.columns, axis=1, inplace=True, errors="ignore"
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
        self.ranked_item_score = list(item_score.sort_values(by="movie_score", ascending=False)["movie"])

    def rec(self):
        for movie in self.picked_userid_watched.columns :
            self.ranked_item_score.extend(list(pd.DataFrame(self.movie_similarity[movie]).sort_values(by=movie, ascending=False).head(5).index))
        for i in self.ranked_item_score:
            print(i)


if __name__ == "__main__":
    userCF = UserCF(1)

    userCF.get_data()
    userCF.calc_sim()
    userCF.narrow_down_item_pool()
    userCF.get_rec()
    userCF.rec()
