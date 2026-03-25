-- Invalidate valid_email when email changes
DELIMITER $$

CREATE TRIGGER UpdateMail
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
    -- Email changed
    IF NEW.email <> OLD.email THEN
        SET NEW.valid_email = 0;
    END IF;
END $$

DELIMITER;