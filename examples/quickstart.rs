use ordvec::RankQuant;

fn main() {
    // Three tiny embeddings with distinct coordinate orderings.
    let documents = [
        8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, // document 0
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // document 1
        8.0, 1.0, 7.0, 2.0, 6.0, 3.0, 5.0, 4.0, // document 2
    ];
    let query = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

    let mut index = RankQuant::new(8, 1);
    index.add(&documents);
    let results = index.search_asymmetric(&query, 1);

    let top_document = results.indices_for_query(0)[0];
    let top_score = results.scores_for_query(0)[0];
    assert_eq!(top_document, 0);
    println!("top document: {top_document} (score {top_score:.3})");
}
