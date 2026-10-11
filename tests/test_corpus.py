from dianoia import corpus

BODY = """I made a few changes.  Generally when drafting a summary, shorter is better
(ideally one page).

Separately, are you free on Tuesday?

-Eric




"Sam Doe" <sam.doe@example.com> on 01/08/2001 09:10:20 AM
To: <pat.roe@example.com>
Subject: Draft summary

Could you take a quick look at the attached draft?"""


def test_authored_stops_at_the_quoted_header():
    text = corpus.authored(BODY)
    assert "attached" not in text
    assert text.endswith("-Eric")


def test_authored_stops_at_original_message():
    assert corpus.authored("Yes.\n\n -----Original Message-----\nFrom: x") == "Yes."


def test_paragraphs_unwrap_hard_wraps():
    assert corpus.paragraphs(corpus.authored(BODY))[0].startswith("I made a few changes. Generally")


def test_thread_key_strips_every_prefix():
    assert corpus.thread_key("RE: Fw: re:  Gas Deal") == "gas deal"


def test_is_prose_rejects_addresses_and_shouting():
    assert not corpus.is_prose("Send it to john.doe@example.com by Friday morning please", min_words=5, max_words=60)
    assert not corpus.is_prose("PLEASE READ THIS VERY IMPORTANT NOTICE NOW TODAY", min_words=5, max_words=60)
    assert corpus.is_prose("We should move the meeting to Thursday afternoon.", min_words=5, max_words=60)
