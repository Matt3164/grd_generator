from grd_generator.logger import configure_logging, logger


def test_configure_logging_is_idempotent():
    configure_logging()
    configure_logging()  # second call must be a no-op, not raise
    logger.info("test_event", key="value")  # must not raise
