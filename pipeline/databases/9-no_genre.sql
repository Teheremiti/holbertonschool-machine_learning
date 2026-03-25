-- Shows with no genres
SELECT shows.title, genres.genre_id
FROM tv_shows AS shows
LEFT JOIN tv_show_genres AS genres
ON shows.id = show_genre.show_id
WHERE genres.genre_id IS NULL
ORDER BY shows.title ASC, genres.genre_id ASC;