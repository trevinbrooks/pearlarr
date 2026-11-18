"""The table Sonarr strips accents from a cleaned series title with.

Sonarr strips a title it has lowercased, so the table holds only the entries a lowercased string can hold: those of
the Diacritical.Net package, and two of Sonarr's own three (`AdditionalDiacriticsProvider`). Its third, the capital
Ð, is left out, since lowercasing turns it into the ð already here.
"""

from typing import Final

# The entries of _PACKAGE are from Diacritical.Net 1.0.5 by Anthony Halliday
# (github.com/anth12/diacritical-dotnet), used under the MIT License:
#
# Copyright (c) 2019 Anthony Halliday
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
_PACKAGE: Final = {
    "A": "ᴀ",
    "a": "àáâãäåāăąǎǟǡǻȁȃȧɐᶏḁẚạảấầẩẫậắằẳẵặₐⱥ",
    "aa": "ꜳ",
    "AE": "ᴁ",
    "ae": "æǣǽᴂ",
    "ao": "ꜵ",
    "au": "ꜷ",
    "av": "ꜹꜻ",
    "ay": "ꜽ",
    "B": "ʙᴃ",
    "b": "ƀƃɓᵬᶀḃḅḇ",
    "C": "ᴄ",
    "c": "çćĉċčƈȼɕḉↄꜿ",
    "D": "ᴅ",
    "d": "ďđƌȡɖɗᵭᶁᶑḋḍḏḑḓꝺ",
    "dz": "ǆǳ",
    "E": "ᴇⱻ",
    "e": "èéêëēĕėęěǝȅȇȩɇɘɛᶒᶓḕḗḙḛḝẹẻẽếềểễệₑⱸ",
    "et": "ꝫ",
    "F": "ꜰ",
    "f": "ƒᵮᶂḟꝼ",
    "ff": "ﬀ",
    "ffi": "ﬃ",
    "ffl": "ﬄ",
    "fi": "ﬁ",
    "fl": "ﬂ",
    "G": "ɢʛ",
    "g": "ĝğġģǥǧǵɠɡᵷᵹᶃḡ",
    "H": "ʜ",
    "h": "ĥħȟɥɦʮʯḣḥḧḩḫẖⱨ",
    "hv": "ƕ",
    "I": "İɪ",
    "i": "ìíîïĩīĭįıǐȉȋɨᴉᵢᶖḭḯỉị",
    "ij": "ĳ",
    "is": "ꝭ",
    "J": "ᴊ",
    "j": "ĵǰȷɉɟʄʝⱼ",
    "K": "ᴋ",
    "k": "ķƙǩʞᶄḱḳḵⱪꝁꝃꝅ",
    "L": "ʟᴌ",
    "l": "ĺļľŀłƚȴɫɬɭᶅḷḹḻḽⱡꝉꞁ",
    "lj": "ǉ",
    "M": "ᴍ",
    "m": "ɯɰɱᵯᶆḿṁṃ",
    "N": "ɴᴎ",
    "n": "ñńņňƞǹȵɲɳᵰᶇṅṇṉṋ",
    "nj": "ǌ",
    "O": "ᴏᴐ",
    "o": "òóôõöøōŏőơǒǫǭǿȍȏȫȭȯȱɔɵᴑᴓᶗṍṏṑṓọỏốồổỗộớờởỡợₒⱺꝋꝍ",
    "OE": "ɶ",
    "oe": "œᴔ",
    "oi": "ƣ",
    "oo": "ꝏ",
    "OU": "ᴕ",
    "ou": "ȣ",
    "P": "ᴘ",
    "p": "ƥᵱᵽᶈṕṗꝑꝓꝕ",
    "q": "ɋʠꝗꝙ",
    "R": "ʀʁᴙᴚ",
    "r": "ŕŗřȑȓɍɹɺɻɼɽɾɿᵣᵲᵳᶉṙṛṝṟⱹꞃ",
    "S": "ꜱ",
    "s": "śŝşšſșȿʂᵴᶊṡṣṥṧṩẛẜẝꞅ",
    "ss": "ß",
    "st": "ﬆ",
    "T": "ᴛ",
    "t": "ţťŧƫƭțȶʇʈᵵṫṭṯṱẗⱦꞇ",
    "th": "ᵺ",
    "tz": "ꜩ",
    "U": "ᴜ",
    "u": "ùúûüũūŭůűųưǔǖǘǚǜȕȗᴝᵤᶙṳṵṷṹṻụủứừửữự",
    "ue": "ᵫ",
    "um": "ꝸ",
    "V": "ᴠ",
    "v": "ʋʌᵥᶌṽṿⱱⱴꝟ",
    "vy": "ꝡ",
    "W": "ᴡ",
    "w": "ŵʍẁẃẅẇẉẘⱳ",
    "x": "ᶍẋẍₓ",
    "Y": "ʏ",
    "y": "ýÿŷƴȳɏʎẏẙỳỵỷỹỿ",
    "Z": "ᴢ",
    "z": "źżžƶȥɀʐʑᵶᶎẑẓẕⱬ",
}
"""Each plain replacement and the characters it replaces. Some are capitals, which Sonarr leaves as they are."""

_SONARR: Final = {"d": "ð", "th": "þ"}

UNACCENTED: Final[dict[int, str]] = {
    ord(char): plain for table in (_PACKAGE, _SONARR) for plain, chars in table.items() for char in chars
}
"""The table as `str.translate` takes it: each accented character's code to its plain replacement."""
