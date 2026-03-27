-- Update user's average_score from corrections
DROP TRIGGER IF EXISTS ComputeAverageScoreForUser;

DELIMITER $$

CREATE PROCEDURE ComputeAverageScoreForUser(
    IN user_id INT)

    BEGIN
        UPDATE users
        SET average_score = (
                SELECT AVG(corrections_table.score)
                FROM corrections AS corrections_table
                WHERE corrections_table.user_id = user_id)
        WHERE id = user_id;

    END $$

DELIMITER;