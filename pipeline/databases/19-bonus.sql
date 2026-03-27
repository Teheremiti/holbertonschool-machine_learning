-- Add bonus correction (stored procedure)
DELIMITER $$

-- Params
CREATE PROCEDURE AddBonus(
    IN user_id INT,
    IN project_name VARCHAR(255),
    IN score INT)

BEGIN
    -- Create project if needed
    IF NOT EXISTS (
        SELECT 1
        FROM projects AS project_table
        WHERE project_table.name=project_name
        ) THEN
            INSERT INTO projects(name)
            VALUES (project_name);
    END IF;

    -- Create correction row
    INSERT INTO corrections (
        user_id,
        project_id,
        score)
        VALUES (
            user_id, (SELECT project_table.id FROM projects AS project_table
                     WHERE project_table.name=project_name),
            score);

END $$
DELIMITER;
