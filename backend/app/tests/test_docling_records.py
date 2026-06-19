from __future__ import annotations

import unittest

from app.utils.docling_records import build_chunk_records, build_item_records


class DoclingRecordsTests(unittest.TestCase):
    def test_build_item_records_keeps_table_captions_out_of_section_path(self) -> None:
        section = self.make_item("#/texts/1", "section_header", "3 Randbedingungen")
        table_1 = self.make_item("#/tables/0", "table", None)
        caption_like_header = self.make_item("#/texts/2", "section_header", "Tab.6 Maximale Nachzerfallsleistung")
        table_2 = self.make_item("#/tables/1", "table", None)
        document = self.make_document([(section, 1), (table_1, 1), (caption_like_header, 1), (table_2, 1)])

        records = build_item_records(document)
        table_2_record = next(record for record in records if record["self_ref"] == "#/tables/1")

        self.assertEqual(table_2_record["section_path"], ["3 Randbedingungen"])

    def test_build_item_records_infers_nearest_asset_caption_on_page(self) -> None:
        caption_far = self.make_item(
            "#/texts/10",
            "section_header",
            "Tab.5 Entfernte Tabelle",
            page_no=22,
            bbox=(50, 700, 300, 680),
        )
        table = self.make_item(
            "#/tables/0",
            "table",
            None,
            page_no=22,
            bbox=(100, 500, 400, 300),
        )
        caption_near = self.make_item(
            "#/texts/11",
            "section_header",
            "Tab.6 Richtige Tabelle",
            page_no=22,
            bbox=(90, 290, 360, 270),
        )
        document = self.make_document([(caption_far, 1), (table, 1), (caption_near, 1)])

        records = build_item_records(document)
        table_record = next(record for record in records if record["self_ref"] == "#/tables/0")

        self.assertEqual(table_record["asset_caption"], "Tab.6 Richtige Tabelle")
        self.assertEqual(table_record["asset_caption_ref"], "#/texts/11")
        self.assertEqual(table_record["asset_caption_source"], "nearest_candidate_on_page")

    def test_build_chunk_records_includes_docling_captions_in_contextualized_text(self) -> None:
        section = self.make_item("#/texts/1", "section_header", "4 Ergebnisse")
        table = self.make_item("#/tables/0", "table", None)
        document = self.make_document([(section, 1), (table, 1)])
        chunk = self.make_chunk(
            [table],
            "0,086 | 0,149 | 1,628",
            captions=["Tab.6 Maximale Nachzerfallsleistung und Referenzwert für den Behälter"],
        )

        records = build_chunk_records(document, chunker=self.make_chunker([chunk]))

        self.assertIn("4 Ergebnisse", records[0]["contextualized_text"])
        self.assertIn("Tab.6 Maximale Nachzerfallsleistung", records[0]["contextualized_text"])
        self.assertIn("0,086 | 0,149", records[0]["contextualized_text"])

    def test_build_chunk_records_accepts_none_docling_headings(self) -> None:
        document = self.make_document([])
        chunk = self.make_chunk([], "scanned text", headings=None)

        records = build_chunk_records(document, chunker=self.make_chunker([chunk]))

        self.assertEqual(records[0]["section_path"], [])
        self.assertEqual(records[0]["contextualized_text"], "scanned text")

    @staticmethod
    def make_item(
        self_ref: str,
        label_value: str,
        text: str | None,
        *,
        parent_ref: str | None = None,
        page_no: int | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ):
        label = type("Label", (), {"value": label_value})()
        parent = type("Parent", (), {"cref": parent_ref})() if parent_ref is not None else None
        prov = []
        if page_no is not None:
            bbox_obj = None
            if bbox is not None:
                bbox_obj = type(
                    "BBox",
                    (),
                    {
                        "l": bbox[0],
                        "t": bbox[1],
                        "r": bbox[2],
                        "b": bbox[3],
                        "coord_origin": "TOPLEFT",
                    },
                )()
            prov = [type("Prov", (), {"page_no": page_no, "bbox": bbox_obj, "charspan": None})()]
        return type(
            "Item",
            (),
            {
                "label": label,
                "self_ref": self_ref,
                "parent": parent,
                "children": [],
                "text": text,
                "orig": text,
                "prov": prov,
                "data": None,
            },
        )()

    @staticmethod
    def make_document(items):
        return type("Document", (), {"iterate_items": lambda self: iter(items)})()

    @staticmethod
    def make_chunk(doc_items, text: str, *, captions: list[str] | None = None, headings: list[str] | None = None):
        meta = type("Meta", (), {"doc_items": doc_items, "captions": captions or [], "headings": headings})()
        return type("Chunk", (), {"meta": meta, "text": text})()

    @staticmethod
    def make_chunker(chunks):
        return type("Chunker", (), {"chunk": lambda self, document: iter(chunks)})()


if __name__ == "__main__":
    unittest.main()
