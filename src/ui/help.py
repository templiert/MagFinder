from src.utils.html import print_list, tag

L_HELP = (
    "[l] exports section transforms. Each transform transforms its section from global to local coordinates. "
    "If you apply the transforms to the global sections then:"
    + print_list(
        "the centroids of all transformed local sections are at (0,0)",
        "the transformed local sections are aligned",
    )
)

HEADER_HELP = (
    "<html><br>[a] = Press the key a<br>[A] = [a] + [shift] = Press the key a while holding [shift] down<br>"
    + '<p style="text-align:center"><a href="https://youtu.be/ZQLTBbM6dMA">20-minute video tutorial</a></p>'
)

HELP_MSG_GLOBAL = (
    HEADER_HELP
    + tag("Navigation", "h3")
    + print_list(
        "Zoom in/out"
        + print_list(
            "[Ctrl] + mouse wheel",
            "[+]/[-]",
        ),
        "Move up/down" + print_list("mouse wheel", "[&uarr] / [&darr]"),
        "Move left/right" + print_list("[Shift] + mouse wheel", "[&larr] / [&rarr]"),
        "Toggle"
        + print_list(
            "[1] filling of sections",
            "[2] filling of rois",
            "[3] filling of focus",
            "[0] display labels",
        ),
    )
    + tag("Action", "h3")
    + print_list(
        "Mouse click"
        + print_list(
            "Left - draws point",
            "Right - joins first and last points to close the current polygon",
        ),
        "[h] opens this help message",
        "[escape] stops current drawing",
        "[a] adds an annotation that you have drawn",
        "[t] toggles to local mode",
        "[q] quits (everything will be saved)",
        "[s] saves (happens automatically when toggling [t] and quitting [q])",
        "[m] exports a summary image of global mode",
        "[n] renumbers the sections to have continuous numbers without gaps (0,1,3,4,6 --> 0,1,2,3,4)",
        "[k] transfers annotations from a source wafer into this currently open target wafer (using landmarks for the transform)",
        "[b] exports global magnet stack given slab ids",
        (
            "[o]   (letter o) computes the section order that minimizes the travel of the microscope stage"
            " (not the same as the serial sectioning order of the sections)."
            ' Saves the order in the .magc file in the field "stage_order"'
        ),
        (
            "[j] computes the serial order based on the roi defined in the first section."
            " Updates the section positions."
            'Stores the serial order in the .magc file in the field "serial_order".'
            " We recommend to save beforehand a copy of the .magc file outside of the directory"
        ),
        "[y] reverses the serial order",
        L_HELP,
    )
    + "<br><br><br></html>"
)
HELP_MSG_LOCAL = (
    HEADER_HELP
    + tag("Navigation", "h3")
    + print_list(
        "Navigates annotations up/down" + print_list("[d]/[f]", "Mouse wheel"),
        "Navigates 10 annotations up/down"
        + print_list("[c]/[v]", "[Ctrl] + Mouse wheel"),
        "[e]/[r] navigate to first/last annotation",
        "[D]/[F], [C]/[V], [E]/[R], same as lowercase commands above but navigates "
        + tag("sections", "u")
        + " instead of annotations",
        "[+]/[-] zoom in/out",
        "If you lose the current annotation (by clicking outside of the annotation), then press [d],[f] or use the mouse wheel to make the annotation appear again.",
    )
    + tag("Action", "h3")
    + print_list(
        "[h] opens this help message",
        "[a] creates/modifies an annotation",
        "[t] toggles to global mode",
        "[x] deletes the currently selected annotation",
        "[p] propagates the current annotation to sections defined in the dialog. Section and landmark annotations cannot be propagated",
        "[g]/[G] propagates all annotations of the current section to the next/previous serial section and moves to the next/previous serial section",
        "[q] quits. Everything will be saved",
        "[s] saves to file (happens already automatically when toggling [t] or quitting [q])",
        "[1]/[2] pushes the currently displayed section backward/forward. It does not recompute alignment.",
        "[m] exports a summary montage and an annotated stack",
        (
            "[o]   (letter o) computes the section order that minimizes the travel of the microscope stage"
            " (not the same as the serial sectioning order of the sections)."
            ' Saves the order in the .magc file in the field "stage_order"'
        ),
        L_HELP,
    )
    + "<br><br><br></html>"
)

MSG_DRAWN_ROI_MISSING = (
    "Please draw something before pressing [a]."
    + "\nAfter closing this message you can press [h] for help."
)
