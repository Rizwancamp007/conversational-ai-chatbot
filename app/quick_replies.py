"""Quick-reply chips shown under the bot's message in the chat UI.

When the bot is waiting on a detail, the chips offer the common values for
it (courses, class levels, subjects). Otherwise they offer the main topics,
so a new visitor can tap instead of typing.
"""

TOPIC_CHIPS = ["Admissions", "Fees", "Courses", "Timetable", "Tests",
               "Results"]

_SLOT_CHIPS = {
    "course": ["FSc", "ICS", "FA", "ICOM"],
    "class_level": ["9th", "10th", "1st year", "2nd year"],
    "subject": ["Physics", "Chemistry", "Maths", "English"],
    # A name can't be suggested, so no chips while waiting for one.
    "student_name": [],
}


def quick_replies(pending_slot=None):
    """Chips for the slot being asked about, or the main topics."""
    if pending_slot:
        return list(_SLOT_CHIPS.get(pending_slot, []))
    return list(TOPIC_CHIPS)
